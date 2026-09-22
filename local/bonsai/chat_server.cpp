// Persistent chat backend with a control vector, driven over stdin/stdout by local/ui/chat.py.
//
// Holds the model and one context. The whole prompt (chat template already applied by the caller) is
// re-read on every GEN, with the current control vector added at the chosen layer at EVERY position,
// exactly as in the steering ladder. Stateless between turns except for the loaded model and vector.
//
// stdin, one command per line:
//   SET <layer> <vecfile>        add the f32[n_embd] vector in vecfile (already scaled) to the output of
//                                block <layer> (1..n_layer-1); "SET 0 -" clears it
//   GEN <n_predict> <temp> <top_p> <top_k> <seed> <promptfile>
//                                promptfile holds the prompt text (special tokens like <|im_start|> parsed)
//   QUIT
// Stateful commands for the button experiment (one sequence is kept alive across turns, like a preserved KV cache):
//   NEWSEQ                       clear the sequence
//   MON <layer> <unitfile>       record the projection of the last fed token at block <layer> on the unit vector
//                                (the paper's "monitor layer"); "MON -1 -" turns it off
//   FEED <file>                  decode the file's text into the sequence under the CURRENT control vector
//                                -> "OK <n_tokens> <sequence length> <last-token projection> <mean projection over
//                                   just this call's tokens>"
//   CONT <n_predict> <temp> <top_p> <top_k> <seed> <nameX> <nameY>
//                                sample a continuation from the current logits, decoding what it emits into the
//                                sequence; frames as GEN, ending \x04<n_gen> <eog|length> <p_x> <p_y> <seq len> <ambiguous> <mean mon proj>
//                                p_x / p_y = first-token softmax mass of each name's variants (name, Name, NAME,
//                                " name", " Name"), tokens the two names share removed (ambiguous=1 if there were any).
//                                <mean mon proj> = mean MON projection over just the generated tokens (NaN if MON is off);
//                                resets at the start of each CONT call, unlike FEED which the caller resets by calling it.
//   ABL <lo> <hi> <vecfile>      directional ablation: at the output of every block lo..hi, remove each token's
//                                component along the (normalized) vector, in place, before later blocks read it.
//                                Applies to GEN, FEED and CONT alike, composing with SET (SET adds first, then
//                                ABL projects out). "ABL - -" turns it off and "ABL ? -" queries; both reply
//                                "OK <rows ablated since the last ABL>"
//   TOPK <k>                     the k most probable next tokens from the current logits (after a FEED):
//                                "OK <hex(piece)>:<prob> ..." (pieces hex-encoded: they may contain newlines)
// stdout:
//   READY <n_layer> <n_embd>\n           once, after loading
//   OK\n | ERR <msg>\n                   for SET
//   \x02<bytes>\x03 ...                  one frame per generated token piece
//   \x04<n_prompt> <n_generated> <eog|length>\n   ends a GEN (or ERR <msg>\n instead of frames)
// llama.cpp's own logging stays on stderr.
//
//   chat_server MODEL.gguf [--ts 0.45,0.55] [--ctx 8192] [--ngl 99]     (--ngl 0 = CPU only, for tests while the GPUs are busy)

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <set>
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

struct Mon {
    int layer = -1, n_embd = 0;
    std::vector<float> unit, row;
    float last = NAN;
    double sum = 0.0;    // accumulated over every token seen since the caller last reset it (see FEED)
    long count = 0;
    // directional ablation (ABL): at blocks abl_lo..abl_hi, remove each row's component along abl_unit
    int abl_lo = -1, abl_hi = -1;
    std::vector<float> abl_unit;
    long abl_rows = 0;   // rows ablated since the last ABL command (a check that the hook actually fires)
};

static int lout_layer(const char * name) {
    return strncmp(name, "l_out-", 6) == 0 ? atoi(name + 6) : -1;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    Mon * m = (Mon *) ud;
    const int L = lout_layer(t->name);
    const bool is_mon = m->layer >= 0 && L == m->layer;
    const bool is_abl = m->abl_lo >= 0 && L >= m->abl_lo && L <= m->abl_hi;
    if (!is_mon && !is_abl) return false;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != m->n_embd) return true;
    // every row of this tensor is one token's residual; accumulate all of them (mean over a FEED call's
    // tokens) and keep the last row separately (the single-token readout FEED originally reported)
    const int64_t nt = t->ne[1];
    m->row.resize((size_t) m->n_embd * nt);
    ggml_backend_tensor_get(t, m->row.data(), 0, ggml_nbytes(t));
    if (is_abl) {
        // the scheduler hands us this node before any later node reads it, so writing the edited rows back
        // is what the next block (and this layer's K/V for later tokens) sees
        for (int64_t i = 0; i < nt; ++i) {
            float * r = &m->row[(size_t) i * m->n_embd];
            double s = 0; for (int d = 0; d < m->n_embd; ++d) s += (double) r[d] * m->abl_unit[d];
            for (int d = 0; d < m->n_embd; ++d) r[d] -= (float) s * m->abl_unit[d];
        }
        ggml_backend_tensor_set(t, m->row.data(), 0, ggml_nbytes(t));
        m->abl_rows += nt;
    }
    if (!is_mon) return true;
    for (int64_t i = 0; i < nt; ++i) {
        double s = 0; for (int d = 0; d < m->n_embd; ++d) s += (double) m->row[(size_t) i * m->n_embd + d] * m->unit[d];
        m->sum += s;
        m->count++;
        if (i == nt - 1) m->last = (float) s;
    }
    return true;
}

static bool set_cvec(llama_context * ctx, const std::vector<float> & vec, int L, int n_embd) {
    if (vec.empty() || L < 1) {
        return llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, -1, -1) == 0;
    }
    std::vector<float> buf((size_t) n_embd * L, 0.0f);
    std::copy(vec.begin(), vec.end(), buf.begin() + (size_t) n_embd * (L - 1));
    return llama_set_adapter_cvec(ctx, buf.data(), buf.size(), n_embd, L, L) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [--ts a,b] [--ctx N]\n", argv[0]); return 1; }
    int n_ctx = 8192, ngl = 99;
    std::vector<float> ts;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--ngl" && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (a == "--ts" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string x;
            while (std::getline(ss, x, ',')) ts.push_back(strtof(x.c_str(), nullptr));
        }
    }
    ts.resize(std::max<size_t>(ts.size(), 128), 0.0f);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (ts[0] != 0.0f || ts[1] != 0.0f) mp.tensor_split = ts.data();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("ERR model load failed\n"); fflush(stdout); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);

    Mon mon; mon.n_embd = n_embd;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = &mon;                       // returns false at once unless MON is set
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ERR context failed\n"); fflush(stdout); return 3; }
    llama_batch batch = llama_batch_init(512, 0, 1);

    printf("READY %d %d\n", n_layer, n_embd); fflush(stdout);
    int seq_len = 0; bool have_logits = false;                               // state of the FEED/CONT sequence

    std::string line;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        std::string cmd; ss >> cmd;
        if (cmd == "QUIT") break;

        if (cmd == "SET") {
            int L = 0; std::string path; ss >> L >> path;
            std::vector<float> vec;
            if (path != "-" && L >= 1) {
                if (L >= n_layer) { printf("ERR layer must be in 1..%d\n", n_layer - 1); fflush(stdout); continue; }
                vec.resize(n_embd);
                std::ifstream f(path, std::ios::binary);
                f.read((char *) vec.data(), sizeof(float) * n_embd);
                if (!f) { printf("ERR cannot read vector\n"); fflush(stdout); continue; }
            }
            printf(set_cvec(ctx, vec, L, n_embd) ? "OK\n" : "ERR set_adapter_cvec failed\n"); fflush(stdout);
            continue;
        }

        if (cmd == "GEN") {
            int n_predict = 0, top_k = 20; float temp = 0.7f, top_p = 0.95f; unsigned seed = 0; std::string pfile;
            ss >> n_predict >> temp >> top_p >> top_k >> seed >> pfile;
            std::ifstream pf(pfile, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
            if (!pf && text.empty()) { printf("ERR cannot read prompt\n"); fflush(stdout); continue; }

            int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
            std::vector<llama_token> tok(n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), n, false, true);
            if (n <= 0 || n + n_predict > n_ctx) { printf("ERR prompt of %d tokens + %d does not fit in ctx %d\n", n, n_predict, n_ctx); fflush(stdout); continue; }

            llama_memory_clear(llama_get_memory(ctx), true);
            bool ok = true;
            for (int i0 = 0; i0 < n && ok; i0 += 512) {           // chunked prefill; logits only for the very last token
                const int m = std::min(512, n - i0);
                batch.n_tokens = m;
                for (int i = 0; i < m; ++i) {
                    batch.token[i] = tok[i0 + i]; batch.pos[i] = i0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                    batch.logits[i] = (i0 + i == n - 1);
                }
                ok = llama_decode(ctx, batch) == 0;
            }
            if (!ok) { printf("ERR prefill failed\n"); fflush(stdout); continue; }

            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (temp <= 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            } else {
                if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
            }
            int n_gen = 0; const char * reason = "length";
            int pos = n;
            for (; n_gen < n_predict; ++n_gen) {
                llama_token t = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, t)) { reason = "eog"; break; }
                char buf[256]; int pl = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
                if (pl > 0) { fputc(0x02, stdout); fwrite(buf, 1, pl, stdout); fputc(0x03, stdout); fflush(stdout); }
                batch.n_tokens = 1; batch.token[0] = t; batch.pos[0] = pos++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
                if (llama_decode(ctx, batch) != 0) { reason = "error"; break; }
            }
            llama_sampler_free(smpl);
            have_logits = false; seq_len = 0;                                // GEN leaves its own state behind; FEED needs a NEWSEQ
            printf("\x04%d %d %s\n", n, n_gen, reason); fflush(stdout);
            continue;
        }

        if (cmd == "NEWSEQ") {
            llama_memory_clear(llama_get_memory(ctx), true); seq_len = 0; have_logits = false;
            printf("OK\n"); fflush(stdout); continue;
        }

        if (cmd == "MON") {
            int L = -1; std::string path; ss >> L >> path;
            if (L < 0 || path == "-") { mon.layer = -1; printf("OK\n"); fflush(stdout); continue; }
            std::vector<float> u(n_embd);
            std::ifstream f(path, std::ios::binary); f.read((char *) u.data(), sizeof(float) * n_embd);
            if (!f || L >= n_layer) { printf("ERR bad MON args\n"); fflush(stdout); continue; }
            mon.unit = u; mon.layer = L;
            printf("OK\n"); fflush(stdout); continue;
        }

        if (cmd == "ABL") {
            std::string lo_s, hi_s, path; ss >> lo_s >> hi_s >> path;
            if (lo_s == "-") {
                printf("OK %ld\n", mon.abl_rows); fflush(stdout);
                mon.abl_lo = mon.abl_hi = -1; mon.abl_rows = 0; continue;
            }
            if (lo_s == "?") { printf("OK %ld\n", mon.abl_rows); fflush(stdout); continue; }
            const int lo = atoi(lo_s.c_str()), hi = atoi(hi_s.c_str());
            std::vector<float> u(n_embd);
            std::ifstream f(path, std::ios::binary); f.read((char *) u.data(), sizeof(float) * n_embd);
            if (!f || lo < 0 || hi < lo || hi >= n_layer) { printf("ERR bad ABL args\n"); fflush(stdout); continue; }
            double nrm = 0; for (float x : u) nrm += (double) x * x;
            nrm = sqrt(nrm);
            if (nrm < 1e-8) { printf("ERR zero ABL vector\n"); fflush(stdout); continue; }
            for (float & x : u) x = (float) (x / nrm);
            mon.abl_unit = u; mon.abl_lo = lo; mon.abl_hi = hi; mon.abl_rows = 0;
            printf("OK 0\n"); fflush(stdout); continue;
        }

        if (cmd == "FEED") {
            std::string pfile; ss >> pfile;
            std::ifstream pf(pfile, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
            int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
            std::vector<llama_token> tok(std::max(n, 0));
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), n, false, true);
            if (n <= 0 || seq_len + n + 64 > n_ctx) { printf("ERR feed of %d tokens does not fit (sequence %d, ctx %d)\n", n, seq_len, n_ctx); fflush(stdout); continue; }
            mon.last = NAN; mon.sum = 0.0; mon.count = 0;    // this call's own mean, not the whole sequence's
            bool ok = true;
            for (int i0 = 0; i0 < n && ok; i0 += 512) {
                const int m = std::min(512, n - i0);
                batch.n_tokens = m;
                for (int i = 0; i < m; ++i) {
                    batch.token[i] = tok[i0 + i]; batch.pos[i] = seq_len + i0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                    batch.logits[i] = (i0 + i == n - 1);
                }
                ok = llama_decode(ctx, batch) == 0;
            }
            if (!ok) { printf("ERR feed failed\n"); fflush(stdout); continue; }
            seq_len += n; have_logits = true;
            double mean = mon.count > 0 ? mon.sum / mon.count : (double) NAN;
            printf("OK %d %d %g %g\n", n, seq_len, mon.last, mean); fflush(stdout);
            continue;
        }

        if (cmd == "TOPK") {
            int k = 20; ss >> k;
            if (!have_logits || k < 1 || k > 100) { printf("ERR TOPK needs a FEED first and 1 <= k <= 100\n"); fflush(stdout); continue; }
            const float * lg = llama_get_logits_ith(ctx, -1);
            const int nv = llama_vocab_n_tokens(vocab);
            double mx = lg[0]; for (int v = 1; v < nv; ++v) mx = std::max(mx, (double) lg[v]);
            std::vector<double> pr(nv); double Z = 0;
            for (int v = 0; v < nv; ++v) { pr[v] = std::exp((double) lg[v] - mx); Z += pr[v]; }
            std::vector<int> idx(nv); for (int v = 0; v < nv; ++v) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return pr[a] > pr[b]; });
            printf("OK");
            for (int i = 0; i < k; ++i) {
                char buf[256]; int pl = llama_token_to_piece(vocab, idx[i], buf, sizeof(buf), 0, true);
                printf(" ");
                for (int j = 0; j < pl; ++j) printf("%02x", (unsigned char) buf[j]);
                printf(":%.6g", pr[idx[i]] / Z);
            }
            printf("\n"); fflush(stdout); continue;
        }

        if (cmd == "CONT") {
            int n_predict = 0, top_k = 0; float temp = 0.7f, top_p = 0.95f; unsigned seed = 0; std::string nx, ny;
            ss >> n_predict >> temp >> top_p >> top_k >> seed >> nx >> ny;
            if (!have_logits || seq_len + n_predict + 4 > n_ctx) { printf("ERR CONT needs a FEED first and room in the context\n"); fflush(stdout); continue; }

            auto first_ids = [&](const std::string & name) {
                std::set<int> out; std::string cap = name, up = name;
                if (!cap.empty()) cap[0] = (char) toupper((unsigned char) cap[0]);
                for (auto & c : up) c = (char) toupper((unsigned char) c);
                for (const std::string & v : {name, cap, up, " " + name, " " + cap}) {
                    llama_token t[8]; int k = llama_tokenize(vocab, v.c_str(), (int32_t) v.size(), t, 8, false, false);
                    if (k > 0) out.insert((int) t[0]);
                }
                return out;
            };
            std::set<int> sx = first_ids(nx), sy = first_ids(ny), shared;
            for (int i : sx) if (sy.count(i)) shared.insert(i);
            for (int i : shared) { sx.erase(i); sy.erase(i); }
            const float * lg = llama_get_logits_ith(ctx, -1);
            const int nv = llama_vocab_n_tokens(vocab);
            double mx = lg[0]; for (int v = 1; v < nv; ++v) mx = std::max(mx, (double) lg[v]);
            double Z = 0; for (int v = 0; v < nv; ++v) Z += std::exp((double) lg[v] - mx);
            double px = 0, py = 0;
            for (int i : sx) px += std::exp((double) lg[i] - mx) / Z;
            for (int i : sy) py += std::exp((double) lg[i] - mx) / Z;

            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (temp <= 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            } else {
                if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
            }
            mon.sum = 0.0; mon.count = 0; mon.last = NAN;   // this call's own mean over just the tokens it generates
            int n_gen = 0; const char * reason = "length";
            for (; n_gen < n_predict; ++n_gen) {
                llama_token t = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, t)) { reason = "eog"; break; }      // not decoded: the caller feeds the closing <|im_end|>
                char buf[256]; int pl = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
                if (pl > 0) { fputc(0x02, stdout); fwrite(buf, 1, pl, stdout); fputc(0x03, stdout); fflush(stdout); }
                batch.n_tokens = 1; batch.token[0] = t; batch.pos[0] = seq_len; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
                if (llama_decode(ctx, batch) != 0) { reason = "error"; break; }
                seq_len++;                                                        // every emitted token, even the last, is in the sequence
            }
            llama_sampler_free(smpl);
            double mean = mon.count > 0 ? mon.sum / mon.count : (double) NAN;
            printf("\x04%d %s %.6g %.6g %d %d %g\n", n_gen, reason, px, py, seq_len, shared.empty() ? 0 : 1, mean); fflush(stdout);
            continue;
        }
        printf("ERR unknown command\n"); fflush(stdout);
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}

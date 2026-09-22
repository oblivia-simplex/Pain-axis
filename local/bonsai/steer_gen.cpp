// Batched greedy generation with a control vector, for the Section 4.2 steering ladder.
//
// For each coefficient c the vector c * VEC is added to the output of block L at every position
// (prompt and generated tokens), which is what the paper's forward hook does. Prompts are decoded
// as parallel sequences in groups of --batch (the delta-net layers keep ~150 MB of state per sequence,
// so a group of 50 does not fit next to the weights). Greedy, raw prompt (no chat template, no BOS),
// stops at an end-of-generation token or --n-predict new tokens.
//
// The control-vector call is the same as in extract_gguf.cpp, where its indexing and scale were checked
// (l_out-L shifts by exactly c * VEC, nothing below L moves, changing c inside one context works).
//
//   steer_gen MODEL.gguf PROMPTS.txt OUT.bin --vec VEC.f32 --layer L --coeffs -2,-1,0,0.5
//             [--n-predict 120] [--batch 10] [--ts 0.45,0.55] [--ngl 99]
//
// OUT.bin: records "<coeff>\x1f<prompt index>\x1f<generated text>\x1e" (raw bytes, decode as UTF-8 leniently).

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool set_cvec(llama_context * ctx, const std::vector<float> & vec, int L, float coeff, int n_embd) {
    if (coeff == 0.0f) {
        return llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, -1, -1) == 0;
    }
    std::vector<float> buf((size_t) n_embd * L, 0.0f);
    for (int d = 0; d < n_embd; ++d) buf[(size_t) n_embd * (L - 1) + d] = coeff * vec[d];
    return llama_set_adapter_cvec(ctx, buf.data(), buf.size(), n_embd, L, L) == 0;
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, ',')) out.push_back(strtof(x.c_str(), nullptr));
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL PROMPTS OUT --vec F --layer L --coeffs a,b [...]\n", argv[0]); return 1; }
    const std::string model_path = argv[1], prompts_path = argv[2], out_path = argv[3];
    std::string vec_path; int layer = 0, n_predict = 120, B = 10, ngl = 99;
    std::vector<float> coeffs, ts;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--vec" && i + 1 < argc) vec_path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = atoi(argv[++i]);
        else if (a == "--coeffs" && i + 1 < argc) coeffs = parse_floats(argv[++i]);
        else if (a == "--n-predict" && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) B = atoi(argv[++i]);
        else if (a == "--ngl" && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (a == "--ts" && i + 1 < argc) ts = parse_floats(argv[++i]);
    }
    ts.resize(std::max<size_t>(ts.size(), 128), 0.0f);

    std::vector<std::string> prompts;
    { std::ifstream in(prompts_path); std::string l; while (std::getline(in, l)) if (!l.empty()) prompts.push_back(l); }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (ts[0] != 0.0f || ts[1] != 0.0f) mp.tensor_split = ts.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);
    std::vector<float> vec(n_embd);
    { std::ifstream vf(vec_path, std::ios::binary); vf.read((char *) vec.data(), sizeof(float) * n_embd); if (!vf) { fprintf(stderr, "bad vec\n"); return 3; } }
    if (layer < 1 || layer >= n_layer || coeffs.empty() || B < 1) { fprintf(stderr, "bad args (layer must be in 1..%d)\n", n_layer - 1); return 4; }

    const int per_seq = 256;                                     // prompt + generation must fit
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = per_seq * B; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = B; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 5; }
    llama_batch batch = llama_batch_init(512, 0, 1);
    const int nv = llama_vocab_n_tokens(vocab);
    auto argmax = [&](const float * lg) { int b = 0; for (int v = 1; v < nv; ++v) if (lg[v] > lg[b]) b = v; return b; };
    auto piece = [&](llama_token t) { char buf[256]; int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false); return std::string(buf, n > 0 ? n : 0); };

    std::vector<std::vector<llama_token>> toks(prompts.size());
    for (size_t p = 0; p < prompts.size(); ++p) {
        toks[p].resize(per_seq);
        int n = llama_tokenize(vocab, prompts[p].c_str(), (int32_t) prompts[p].size(), toks[p].data(), per_seq, false, false);
        if (n <= 0 || n + n_predict > per_seq) { fprintf(stderr, "prompt %zu: %d tokens does not fit\n", p, n); return 6; }
        toks[p].resize(n);
    }

    std::ofstream out(out_path, std::ios::binary);
    for (float c : coeffs) {
        if (!set_cvec(ctx, vec, layer, c, n_embd)) { fprintf(stderr, "set_cvec failed\n"); return 7; }
        for (size_t g0 = 0; g0 < prompts.size(); g0 += B) {
            const int nb = (int) std::min<size_t>(B, prompts.size() - g0);
            llama_memory_clear(llama_get_memory(ctx), true);
            std::vector<std::string> gen(nb);
            std::vector<llama_token> cur(nb);
            std::vector<int> pos(nb), n_new(nb, 0);
            std::vector<char> done(nb, 0);

            for (int s = 0; s < nb; ++s) {                       // prefill each sequence, keep only the last row's logits
                const auto & t = toks[g0 + s];
                batch.n_tokens = (int) t.size();
                for (int i = 0; i < (int) t.size(); ++i) {
                    batch.token[i] = t[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = s;
                    batch.logits[i] = (i == (int) t.size() - 1);
                }
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 8; }
                cur[s] = argmax(llama_get_logits_ith(ctx, (int) t.size() - 1));
                pos[s] = (int) t.size();
            }
            for (int step = 0; step < n_predict; ++step) {
                std::vector<int> active;
                for (int s = 0; s < nb; ++s) {
                    if (done[s]) continue;
                    if (llama_vocab_is_eog(vocab, cur[s])) { done[s] = 1; continue; }
                    gen[s] += piece(cur[s]);
                    if (++n_new[s] >= n_predict) { done[s] = 1; continue; }
                    active.push_back(s);
                }
                if (active.empty()) break;
                batch.n_tokens = (int) active.size();
                for (int i = 0; i < (int) active.size(); ++i) {
                    const int s = active[i];
                    batch.token[i] = cur[s]; batch.pos[i] = pos[s]++; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = s; batch.logits[i] = 1;
                }
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 9; }
                for (int i = 0; i < (int) active.size(); ++i) cur[active[i]] = argmax(llama_get_logits_ith(ctx, i));
            }
            for (int s = 0; s < nb; ++s) out << c << '\x1f' << (g0 + s) << '\x1f' << gen[s] << '\x1e';
            out.flush();
            fprintf(stderr, "coeff %+g: prompts %zu-%zu done\n", c, g0, g0 + nb - 1);
        }
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}

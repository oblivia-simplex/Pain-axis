// Residual-stream extraction from a GGUF model through llama.cpp's eval callback.
//
// For every prompt (one per line) this captures "l_out-<il>" for every layer, i.e. the residual
// stream after block il (the same point the paper takes with hook_resid_post), and writes
//   final.f32  [n_prompts, n_layers, n_embd]  activation at the last token
//   mean.f32   [n_prompts, n_layers, n_embd]  mean over all tokens of the prompt
//   argmax.txt greedy next token of the last position, one per prompt (sanity check)
//   meta.json  sizes
//
// Built against an existing llama.cpp build (headers + libllama); it does not modify that tree.
//
//   extract_gguf MODEL.gguf PROMPTS.txt OUTDIR [--ngl 99] [--ts 0.45,0.55] [--ctx 512]
//                [--cvec VEC.f32 --cvec-layer L --cvec-coeffs 0,1,2] [--all-tokens]
//
// --all-tokens additionally writes alltok.f32 = [prompt][token][layer][n_embd] (every token at every layer)
// and tokens.txt (one line per prompt, token pieces separated by \x1f); meant for a handful of prompts.
//
// With --cvec the whole prompt set is run once per coefficient in ONE context (changing the control
// vector between passes, as the steering tool does) and written to final_<k>.f32 / mean_<k>.f32.
// This exists to verify the steering path: the difference between passes must be exactly
// coeff * VEC at l_out-L and zero below L.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

struct Cap {
    int n_embd = 0;
    int n_tokens = 0;                         // tokens in the batch being decoded
    std::vector<std::vector<float>> final_;   // [layer][n_embd]
    std::vector<std::vector<float>> mean_;    // [layer][n_embd]
    std::vector<char> seen;
    std::vector<float> buf;
    int n_layers_seen = 0;
    int bad_shape = 0;
    bool keep_all = false;
    std::vector<std::vector<float>> all_;     // [layer][token * n_embd + d], only with keep_all
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    if (strncmp(t->name, "l_out-", 6) != 0) {
        return false;                         // not interested; the callback is only re-called for wanted tensors
    }
    if (ask) {
        return true;
    }
    Cap * c = (Cap *) ud;
    const int il = atoi(t->name + 6);
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || t->ne[1] != c->n_tokens ||
        il < 0 || il >= (int) c->final_.size()) {
        c->bad_shape++;
        return true;
    }
    const int64_t n = t->ne[1];
    c->buf.resize((size_t) c->n_embd * n);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    const float * last = c->buf.data() + (size_t) c->n_embd * (n - 1);
    std::vector<float> & f = c->final_[il];
    std::vector<float> & m = c->mean_[il];
    if (c->keep_all) c->all_[il] = c->buf;
    f.assign(last, last + c->n_embd);
    m.assign(c->n_embd, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        const float * row = c->buf.data() + (size_t) c->n_embd * i;
        for (int d = 0; d < c->n_embd; ++d) m[d] += row[d];
    }
    for (int d = 0; d < c->n_embd; ++d) m[d] /= (float) n;
    if (!c->seen[il]) { c->seen[il] = 1; c->n_layers_seen++; }
    return true;
}

// Adds coeff * vec to the output of block L (0-based, L >= 1) at every position. coeff == 0 clears it.
static bool set_cvec(llama_context * ctx, const std::vector<float> & vec, int L, float coeff, int n_embd) {
    if (coeff == 0.0f) {
        return llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, -1, -1) == 0;
    }
    // data starts at layer 1, so layer L's slice is at n_embd * (L - 1); the buffer ends there
    std::vector<float> buf((size_t) n_embd * L, 0.0f);
    for (int d = 0; d < n_embd; ++d) buf[(size_t) n_embd * (L - 1) + d] = coeff * vec[d];
    return llama_set_adapter_cvec(ctx, buf.data(), buf.size(), n_embd, L, L) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPTS.txt OUTDIR [--ngl N] [--ts a,b] [--ctx N]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1], prompts_path = argv[2], outdir = argv[3];
    int ngl = 99, n_ctx = 512;
    std::vector<float> ts;
    bool all_tokens = false; std::string cvec_path; int cvec_layer = 0; std::vector<float> cvec_coeffs;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ngl" && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--all-tokens") all_tokens = true;
        else if (a == "--cvec" && i + 1 < argc) cvec_path = argv[++i];
        else if (a == "--cvec-layer" && i + 1 < argc) cvec_layer = atoi(argv[++i]);
        else if (a == "--cvec-coeffs" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string x;
            while (std::getline(ss, x, ',')) cvec_coeffs.push_back(strtof(x.c_str(), nullptr));
        }
        else if (a == "--ts" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string x;
            while (std::getline(ss, x, ',')) ts.push_back(strtof(x.c_str(), nullptr));
        }
    }
    ts.resize(std::max<size_t>(ts.size(), 128), 0.0f);   // llama.cpp reads llama_max_devices() entries

    std::vector<std::string> prompts;
    { std::ifstream in(prompts_path); std::string l; while (std::getline(in, l)) if (!l.empty()) prompts.push_back(l); }
    fprintf(stderr, "%zu prompts\n", prompts.size());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (ts[0] != 0.0f || ts[1] != 0.0f) mp.tensor_split = ts.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);
    fprintf(stderr, "n_layer=%d n_embd=%d\n", n_layer, n_embd);

    Cap cap;
    cap.n_embd = n_embd;
    cap.final_.resize(n_layer); cap.mean_.resize(n_layer); cap.seen.assign(n_layer, 0);
    cap.keep_all = all_tokens; cap.all_.resize(n_layer);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.no_perf = true;
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 3; }

    std::vector<float> cvec;
    if (!cvec_path.empty()) {
        cvec.resize(n_embd);
        std::ifstream vf(cvec_path, std::ios::binary);
        vf.read((char *) cvec.data(), sizeof(float) * n_embd);
        if (!vf || cvec_layer < 1 || cvec_layer >= n_layer || cvec_coeffs.empty()) { fprintf(stderr, "bad --cvec args\n"); return 8; }
    } else {
        cvec_coeffs = {0.0f};
    }

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    int n_layers_reported = -1;
    for (size_t k = 0; k < cvec_coeffs.size(); ++k) {
        if (!cvec.empty() && !set_cvec(ctx, cvec, cvec_layer, cvec_coeffs[k], n_embd)) { fprintf(stderr, "set_cvec failed\n"); return 9; }
        const std::string sfx = cvec.empty() ? "" : "_" + std::to_string(k);
        std::ofstream fo(outdir + "/final" + sfx + ".f32", std::ios::binary), mo(outdir + "/mean" + sfx + ".f32", std::ios::binary);
        std::ofstream ao(outdir + "/argmax" + sfx + ".txt");
        std::ofstream to, tk;
        if (all_tokens) { to.open(outdir + "/alltok.f32", std::ios::binary); tk.open(outdir + "/tokens.txt"); }
        if (!fo || !mo || !ao) { fprintf(stderr, "cannot write into %s (create it first)\n", outdir.c_str()); return 4; }
        for (size_t p = 0; p < prompts.size(); ++p) {
            std::vector<llama_token> tok(n_ctx);
            // no BOS/special handling: this model's GGUF has add_bos_token=false, like the HF tokenizer config
            int n = llama_tokenize(vocab, prompts[p].c_str(), (int32_t) prompts[p].size(), tok.data(), n_ctx, false, false);
            if (n <= 0) { fprintf(stderr, "tokenize failed on prompt %zu\n", p); return 5; }

            llama_memory_clear(llama_get_memory(ctx), true);        // reset KV and the recurrent (delta-net) state
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;                                 // keep every row: no output pruning in the last layer
            }
            cap.n_tokens = n; cap.n_layers_seen = 0; std::fill(cap.seen.begin(), cap.seen.end(), 0);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed on prompt %zu\n", p); return 6; }
            if (cap.n_layers_seen != n_layer || cap.bad_shape) {
                fprintf(stderr, "prompt %zu: captured %d/%d layers, %d bad shapes\n", p, cap.n_layers_seen, n_layer, cap.bad_shape);
                return 7;
            }
            for (int l = 0; l < n_layer; ++l) {
                fo.write((const char *) cap.final_[l].data(), sizeof(float) * n_embd);
                mo.write((const char *) cap.mean_[l].data(), sizeof(float) * n_embd);
            }
            if (all_tokens) {
                std::vector<float> row((size_t) n_layer * n_embd);
                for (int t = 0; t < n; ++t) {
                    for (int l = 0; l < n_layer; ++l)
                        std::copy(cap.all_[l].begin() + (size_t) t * n_embd, cap.all_[l].begin() + (size_t) (t + 1) * n_embd, row.begin() + (size_t) l * n_embd);
                    to.write((const char *) row.data(), sizeof(float) * row.size());
                    char pc[128]; int pl2 = llama_token_to_piece(vocab, tok[t], pc, sizeof(pc), 0, true);
                    tk << std::string(pc, pl2 > 0 ? pl2 : 0) << (t + 1 < n ? "\x1f" : "");
                }
                tk << "\n";
            }
            const float * lg = llama_get_logits_ith(ctx, n - 1);
            const int nv = llama_vocab_n_tokens(vocab);
            int best = 0; for (int v = 1; v < nv; ++v) if (lg[v] > lg[best]) best = v;
            char piece[128]; int pl = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
            ao << best << "\t" << std::string(piece, pl > 0 ? pl : 0) << "\t" << n << "\n";
            if (p % 100 == 0) fprintf(stderr, "  pass %zu: %zu/%zu\n", k, p, prompts.size());
            n_layers_reported = n_layer;
        }
    }
    std::ofstream(outdir + "/meta.json") << "{\"n_prompts\": " << prompts.size() << ", \"n_layers\": " << n_layers_reported
                                        << ", \"n_embd\": " << n_embd << ", \"n_passes\": " << cvec_coeffs.size() << "}\n";
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

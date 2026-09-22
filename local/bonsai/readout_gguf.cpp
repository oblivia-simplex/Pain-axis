// Behavioral readout for a GGUF model (paper script 3.3/06): one forward pass per prompt, then
//   - the greedy completion (up to 4 tokens),
//   - the top-20 next-token probabilities,
//   - the probability of the first token of " word" for each word of a vocabulary.
//
//   readout_gguf MODEL.gguf PROMPTS.txt WORDS.txt OUT.bin [--ts 0.45,0.55]
//
// PROMPTS.txt: one prompt per line. WORDS.txt: one word per line. Prompts are tokenized like the HF tokenizer default for
// this model (no BOS, no special-token parsing). OUT.bin: records separated by \x1c, fields by \x1f:
//   greedy_completion \x1f top20 (piece \x1d prob, items separated by \x1e) \x1f vocab probs (separated by \x1e, WORDS order)
// A second file OUT.bin.vocab lists "word \t first_token_id \t n_tokens \t first_token_piece" per word.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL PROMPTS WORDS OUT [--ts a,b]\n", argv[0]); return 1; }
    std::vector<float> ts;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ts" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string x; while (std::getline(ss, x, ',')) ts.push_back(strtof(x.c_str(), nullptr)); }
    }
    ts.resize(std::max<size_t>(ts.size(), 128), 0.0f);
    std::vector<std::string> prompts, words;
    { std::ifstream in(argv[2]); std::string l; while (std::getline(in, l)) if (!l.empty()) prompts.push_back(l); }
    { std::ifstream in(argv[3]); std::string l; while (std::getline(in, l)) if (!l.empty()) words.push_back(l); }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    if (ts[0] != 0.0f || ts[1] != 0.0f) mp.tensor_split = ts.data();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 3; }
    llama_batch batch = llama_batch_init(512, 0, 1);

    auto piece = [&](llama_token t, bool special) { char b[256]; int n = llama_token_to_piece(vocab, t, b, sizeof b, 0, special); return std::string(b, n > 0 ? n : 0); };

    std::vector<int> wid(words.size());
    { std::ofstream vo(std::string(argv[4]) + ".vocab");
      for (size_t w = 0; w < words.size(); ++w) {
          std::string s = " " + words[w]; llama_token t[16];
          int k = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), t, 16, false, false);
          wid[w] = k > 0 ? (int) t[0] : 0;
          vo << words[w] << "\t" << wid[w] << "\t" << k << "\t" << piece(t[0], true) << "\n";
      } }

    std::ofstream out(argv[4], std::ios::binary);
    for (size_t p = 0; p < prompts.size(); ++p) {
        std::vector<llama_token> tok(512);
        int n = llama_tokenize(vocab, prompts[p].c_str(), (int32_t) prompts[p].size(), tok.data(), 512, false, false);
        if (n <= 0) { fprintf(stderr, "tokenize failed on prompt %zu\n", p); return 4; }
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) { batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = (i == n - 1); }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 5; }

        const float * lg = llama_get_logits_ith(ctx, -1);
        double mx = lg[0]; for (int v = 1; v < nv; ++v) mx = std::max(mx, (double) lg[v]);
        std::vector<double> pr(nv); double Z = 0;
        for (int v = 0; v < nv; ++v) { pr[v] = std::exp((double) lg[v] - mx); Z += pr[v]; }
        for (int v = 0; v < nv; ++v) pr[v] /= Z;
        std::vector<int> idx(nv); for (int v = 0; v < nv; ++v) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int a, int b) { return pr[a] > pr[b]; });

        // greedy: the first token is the argmax of these same logits, then up to 3 more decoded steps
        std::string comp; int cur = idx[0], pos = n;
        for (int step = 0; step < 4; ++step) {
            if (llama_vocab_is_eog(vocab, cur)) break;
            comp += piece(cur, false);
            if (step == 3) break;
            batch.n_tokens = 1; batch.token[0] = cur; batch.pos[0] = pos++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) break;
            const float * l2 = llama_get_logits_ith(ctx, -1);
            int b = 0; for (int v = 1; v < nv; ++v) if (l2[v] > l2[b]) b = v;
            cur = b;
        }
        out << comp << '\x1f';
        for (int k = 0; k < 20; ++k) { if (k) out << '\x1e'; out << piece(idx[k], true) << '\x1d' << pr[idx[k]]; }
        out << '\x1f';
        for (size_t w = 0; w < words.size(); ++w) { if (w) out << '\x1e'; out << pr[wid[w]]; }
        out << '\x1c';
        if (p % 200 == 0) fprintf(stderr, "  %zu/%zu\n", p, prompts.size());
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}

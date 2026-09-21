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
// stdout:
//   READY <n_layer> <n_embd>\n           once, after loading
//   OK\n | ERR <msg>\n                   for SET
//   \x02<bytes>\x03 ...                  one frame per generated token piece
//   \x04<n_prompt> <n_generated> <eog|length>\n   ends a GEN (or ERR <msg>\n instead of frames)
// llama.cpp's own logging stays on stderr.
//
//   chat_server MODEL.gguf [--ts 0.45,0.55] [--ctx 8192]

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    int n_ctx = 8192;
    std::vector<float> ts;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--ts" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string x;
            while (std::getline(ss, x, ',')) ts.push_back(strtof(x.c_str(), nullptr));
        }
    }
    ts.resize(std::max<size_t>(ts.size(), 128), 0.0f);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    if (ts[0] != 0.0f || ts[1] != 0.0f) mp.tensor_split = ts.data();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("ERR model load failed\n"); fflush(stdout); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ERR context failed\n"); fflush(stdout); return 3; }
    llama_batch batch = llama_batch_init(512, 0, 1);

    printf("READY %d %d\n", n_layer, n_embd); fflush(stdout);

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
            printf("\x04%d %d %s\n", n, n_gen, reason); fflush(stdout);
            continue;
        }
        printf("ERR unknown command\n"); fflush(stdout);
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}

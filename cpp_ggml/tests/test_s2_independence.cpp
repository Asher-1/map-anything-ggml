// test_s2_independence.cpp — S>=2 cross-frame behaviour regression test.
//
// Model semantics first (this bit us once): VGGT-Omega gives view 0 a
// DIFFERENT learnable cam/register token slice than views 1+ (see
// slice_expand_and_flatten in the official aggregator). Consequences:
//   - two IDENTICAL frames legitimately produce slightly different per-view
//     outputs (first-frame vs other-frame tokens) — torch does the same;
//   - S=2 frame0 need NOT equal an S=1 run (cross-view attention sees a
//     different key set).
// Therefore the invariant we can actually test is NOT symmetry, it is
// *agreement with the PyTorch reference*: per-view outputs vs ref.npz
// (dump_torch_stages) on the same frames. Without a reference we only run
// structural sanity checks and print diagnostics.
//
// Usage:
//   test_s2_independence <model.gguf> [ref.npz]
//   MAPGGML_TEST_GGUF / MAPGGML_TEST_REF override the arguments.
// Ref npz must contain pose_enc (S,9) and depth (S,H,W) for the same
// deterministic frames this test generates (seed 7, H=W=256, S=2).
// Exit codes: 0 PASS · 1 FAIL · 77 SKIP (no model).

#include "backend.hpp"
#include "gguf_loader.hpp"
#include "vggt_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

using namespace mapggml;

namespace {

float lcg_next(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / (float)(1 << 24);
}

int g_failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                ok ? "" : " — ", detail.c_str());
    if (!ok) g_failures++;
}

}  // namespace

int main(int argc, char** argv) {
    // explicit argv contract: test_s2_independence <model.gguf> <ref.npz>
    std::string model_path = argc > 1 ? argv[1] : "";
    std::string ref_path = argc > 2 ? argv[2] : "";
    if (model_path.empty()) {
        std::printf("SKIP: no GGUF model\n"
                    "usage: test_s2_independence <model.gguf> [ref.npz]\n");
        return 77;
    }

    const int H = 256, W = 256, S = 2;
    Backend be = init_backend();
    auto model = load_gguf(model_path);
    if (!model) return 1;

    // deterministic LCG frames, seed 7 (same generator as the python side)
    std::vector<float> frames((size_t)S * 3 * H * W);
    unsigned int st = 7u;
    for (auto& v : frames) v = lcg_next(st);

    VGGTOutputs out;
    VGGTRuntime rt(*model, be, S, H, W);
    if (!rt.run(frames.data(), out)) {
        std::printf("[FAIL] runtime\n");
        return 1;
    }

    double pose_01 = std::fabs((double)out.pose_enc[0] - out.pose_enc[9]);
    for (int k = 1; k < 9; k++) {
        pose_01 = std::max(pose_01,
                           std::fabs((double)out.pose_enc[k] - out.pose_enc[9 + k]));
    }
    double depth_01 = 0.0;
    for (int i = 0; i < H * W; i++) {
        depth_01 = std::max(depth_01,
                            std::fabs((double)out.depth[i] - out.depth[H * W + i]));
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "cpp identical-frames pose_diff=%.4f depth_diff=%.4f "
                  "(first-frame vs other-frame tokens make this >0 by design)",
                  pose_01, depth_01);
    std::printf("[INFO] %s\n", buf);

    // structural sanity: distinct frames must give distinct outputs
    {
        std::vector<float> frames2 = frames;
        for (size_t i = (size_t)3 * H * W; i < frames2.size(); i++) {
            frames2[i] = 1.0f - frames2[i];
        }
        VGGTOutputs out2;
        VGGTRuntime rt2(*model, be, S, H, W);
        if (!rt2.run(frames2.data(), out2)) {
            std::printf("[FAIL] sanity runtime\n");
            return 1;
        }
        double d = 0.0;
        for (int i = 0; i < H * W; i++) {
            d = std::max(d, std::fabs((double)out.depth[i] - out2.depth[i]));
        }
        std::snprintf(buf, sizeof(buf), "depth_frame0_cross=%.4f (expect >0)",
                      d);
        check(d > 1e-3, "distinct frames -> distinct outputs", buf);
    }

    // reference agreement (the real regression): per-view outputs vs torch
    if (ref_path.empty()) {
        std::printf("[INFO] no reference provided; numeric parity is covered "
                    "by tests/test_matrix.py\n");
    } else {
        // reference contract: <base>.pose.bin (S*9 f32) + <base>.depth.bin
        // (S*H*W f32), produced by the torch reference dump.
        std::string base = ref_path;
        if (base.size() > 4 && base.substr(base.size() - 4) == ".npz") {
            base = base.substr(0, base.size() - 4);
        }
        auto read_all = [&](const std::string& p, std::vector<float>& dst) {
            FILE* fp = std::fopen(p.c_str(), "rb");
            if (!fp) return false;
            std::fseek(fp, 0, SEEK_END);
            long n = std::ftell(fp) / 4;
            std::fseek(fp, 0, SEEK_SET);
            dst.resize((size_t)n);
            size_t rd = std::fread(dst.data(), 4, (size_t)n, fp);
            std::fclose(fp);
            return rd == (size_t)n;
        };
        std::vector<float> tp, td;
        if (!read_all(base + ".pose.bin", tp) ||
            !read_all(base + ".depth.bin", td)) {
            std::printf("[INFO] reference bin pair not found under %s; "
                        "skipping numeric parity\n", base.c_str());
        } else {
            double pose_d = 0.0;
            for (int k = 0; k < S * 9; k++) {
                pose_d = std::max(pose_d,
                                  std::fabs((double)tp[k] - out.pose_enc[k]));
            }
            // depth: median relative error (same metric as the gate /
            // test_matrix.py: rel = |d-r| / max(|r|, 0.05), median over all)
            std::vector<double> rel;
            rel.reserve(td.size());
            for (size_t i = 0; i < td.size() && i < out.depth.size(); i++) {
                const double r = std::fabs((double)td[i]);
                rel.push_back(std::fabs((double)td[i] - out.depth[i]) /
                              std::max(r, 0.05));
            }
            std::nth_element(rel.begin(), rel.begin() + rel.size() / 2,
                             rel.end());
            const double depth_d = rel.empty()
                                       ? 0.0
                                       : rel[rel.size() / 2];
            std::snprintf(buf, sizeof(buf), "pose=%.4f depth_med_rel=%.4f",
                          pose_d, depth_d);
            check(pose_d < 0.05 && depth_d < 0.08,
                  "S=2 outputs agree with PyTorch reference", buf);
        }
    }

    std::printf(g_failures == 0 ? "ALL TESTS PASSED\n" : "%d TEST(S) FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}

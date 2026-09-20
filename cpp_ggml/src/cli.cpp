// vggt-cli: run VGGT-Omega inference from GGUF weights.
//
// Parity path (bit-exact preprocessing):
//   vggt-cli --model m.gguf --bin frames.bin --H 294 --W 518 --S 3 \
//            --out-prefix /tmp/vggt
//   (frames.bin = S*3*H*W float32 in [0,1], torch (S,3,H,W) order, produced
//    by the official Python loader)
//
// Image path (official "balanced" preprocessing; H/W come out as
// patch-multiples close to image-resolution^2 tokens):
//   vggt-cli --model m.gguf --images a.jpg b.jpg --image-size 512 \
//            --out-prefix /tmp/vggt
//
// --timing prints a JSON phase breakdown to stderr (load / build /
// stage1..3 / total) and mirrors it into <prefix>.meta.json.
//
// Outputs (three contract families, dispatched by the GGUF architecture —
// see FEATURE_PARITY_AUDIT.md for the family tree):
//   vggt family (vggt-omega / vggt-1b): <prefix>.pose.bin (S*9 f32,
//     pose_enc), .depth.bin, .depth_conf.bin (+ .text_embedding.bin when
//     the GGUF carries the omega TextAlignmentHead weights)
//   pi3 family (pi3/pi3x/mapanything): .pose.bin = row-major c2w 4x4 stack
//     (translation in flat 3/7/11!), .pose_raw.bin, .local_points.bin,
//     .conf.bin, .depth.bin (= local_points camera-z, emitted for a uniform
//     cross-family contract), .points.bin (+ mapanything: .mask.bin,
//     .scale.bin)
//   dust3r family (pair-wise, no pose head): .local_points.bin (both views
//     in view1's frame), .conf.bin, .depth.bin, .points.bin
// plus <prefix>.meta.json for every family.

#include "backend.hpp"
#include "common.hpp"
#include "gguf_loader.hpp"
#include "image_io.hpp"
#include "vggt_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mapggml;

namespace {

double steady_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, bin_path, out_prefix = "/tmp/vggt";
    std::vector<std::string> images;
    int H = 0, W = 0, S = 0;
    int image_size = 512;  // official demo default (per-checkpoint resolution)
    std::string resize_mode = "balanced";  // official default ("balanced"|"max_size")
    int repeats = 1, warmup = 0, cpu_threads = 0;
    bool timing = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                log_error("%s requires a value", what);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--model") model_path = next("--model");
        else if (a == "--bin") bin_path = next("--bin");
        else if (a == "--images") { images.push_back(next("--images")); S = (int)images.size(); }
        else if (a == "--H") H = std::stoi(next("--H"));
        else if (a == "--W") W = std::stoi(next("--W"));
        else if (a == "--S") S = std::stoi(next("--S"));
        else if (a == "--image-size") image_size = std::stoi(next("--image-size"));
        else if (a == "--resize-mode") resize_mode = next("--resize-mode");
        else if (a == "--out-prefix") out_prefix = next("--out-prefix");
        else if (a == "--timing") timing = true;
        else if (a == "--threads") cpu_threads = std::stoi(next("--threads"));
        else if (a == "--repeats") repeats = std::stoi(next("--repeats"));
        else if (a == "--warmup") warmup = std::stoi(next("--warmup"));
        else if (a == "--help" || a == "-h") {
            fprintf(stderr,
                    "usage: vggt-cli --model m.gguf (--bin frames.bin --H h "
                    "--W w --S n | --images a.jpg [b.jpg ...])\n"
                    "                [--image-size 512] [--resize-mode balanced|max_size] "
                    "[--out-prefix p] "
                    "[--timing] [--warmup n] [--repeats n] [--threads n]\n");
            return 0;
        }
        else { log_error("unknown arg: %s", a.c_str()); return 1; }
    }
    if (model_path.empty() || (bin_path.empty() && images.empty())) {
        fprintf(stderr,
                "usage: vggt-cli --model m.gguf (--bin frames.bin --H h --W w "
                "--S n | --images a.jpg [b.jpg ...]) [--image-size 512] "
                "[--resize-mode balanced|max_size] [--out-prefix p] [--timing] "
                "[--warmup n] [--repeats n]\n");
        return 1;
    }

    const double t0 = steady_ms();
    Backend be = init_backend(cpu_threads);
    log_info("backend: %s", be.name.c_str());

    auto model = load_gguf(model_path);
    if (!model) return 1;
    const double t1 = steady_ms();

    std::vector<float> imgs;
    if (!bin_path.empty()) {
        if (H <= 0 || W <= 0 || S <= 0) {
            log_error("--bin requires --H/--W/--S");
            return 1;
        }
        std::vector<uint8_t> raw = read_file_bytes(bin_path);
        const size_t want = (size_t)S * 3 * H * W * sizeof(float);
        if (raw.size() != want) {
            log_error("bin size %zu != expected %zu", raw.size(), want);
            return 1;
        }
        imgs.resize(want / sizeof(float));
        memcpy(imgs.data(), raw.data(), want);
    } else {
        // official load_and_preprocess_images(mode="balanced") equivalent;
        // the patch size comes from the GGUF so any checkpoint works
        if (!load_images_official(images, image_size,
                                  model->meta.patch_size, imgs, H, W,
                                  resize_mode)) {
            return 1;
        }
        S = (int)images.size();
        log_info("preprocessed %d image(s) -> %dx%d (balanced, res=%d, patch=%d)",
                 S, W, H, image_size, model->meta.patch_size);
    }
    const double t2 = steady_ms();

    VGGTOutputs out;
    VGGTRuntime rt(*model, be, S, H, W);
    // Warmup iterations let ggml's CUDA graph replay engage: the CUDA backend
    // only starts capturing after two consecutive computes with unchanged
    // graph properties, and a fresh process would otherwise never reach
    // steady state (single-shot runs pay ~60 ms of kernel-launch overhead).
    // Timed repeats then report per-iteration latency from this same process.
    for (int i = 0; i < warmup; i++) {
        if (!rt.run(imgs.data(), out)) return 1;
    }
    std::vector<double> iter_ms, iter_s1, iter_s2;
    for (int i = 0; i < repeats; i++) {
        if (!rt.run(imgs.data(), out)) return 1;
        iter_ms.push_back(out.timing_ms["inference_total"]);
        if (out.timing_ms.count("stage1_backbone"))
            iter_s1.push_back(out.timing_ms["stage1_backbone"]);
        if (out.timing_ms.count("stage23_agg_heads"))
            iter_s2.push_back(out.timing_ms["stage23_agg_heads"]);
    }
    const double t3 = steady_ms();

    auto dump = [&](const std::vector<float>& v, const std::string& suffix) {
        const std::string path = out_prefix + suffix;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) {
            log_error("cannot write %s", path.c_str());
            return;
        }
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
        log_info("wrote %s (%zu floats)", path.c_str(), v.size());
    };
    if (model->meta.architecture == "dust3r") {
        // pair-wise family (M5): the model has NO pose head (dust3r's
        // poses come from the out-of-network global-alignment optimizer,
        // not ported). Both pointmaps live in view1's frame: s0 = head1
        // pts3d_in_self_view, s1 = head2 pts3d_in_other_view; conf is
        // 1+exp; depth is the pts3d camera-frame z.
        dump(out.local_points, ".local_points.bin");
        dump(out.conf, ".conf.bin");
        dump(out.depth, ".depth.bin");
        dump(out.world_points, ".points.bin");
    } else if (out.camera_poses.empty()) {
        // vggt family contract: pose_enc(S,9) + dedicated depth head.
        // vggt-1b additionally fills world_points/world_points_conf (the
        // official point head, view0-frame world pointmap); vggt-omega has
        // no point head so the conditional dumps are skipped there.
        dump(out.pose_enc, ".pose.bin");
        dump(out.depth, ".depth.bin");
        dump(out.depth_conf, ".depth_conf.bin");
        if (!out.text_embedding.empty())
            dump(out.text_embedding, ".text_embedding.bin");
        if (!out.world_points.empty()) {
            dump(out.world_points, ".points.bin");
            dump(out.world_points_conf, ".points_conf.bin");
        }
    } else {
        // pi3/mapanything family: pose.bin is the camera-to-world 4x4 stack.
        // These models have no dedicated depth head; the official convention
        // (and every eval here) uses the camera-frame z of the local points,
        // so emit .depth.bin too for a uniform cross-family output contract.
        dump(out.camera_poses, ".pose.bin");
        dump(out.pose_raw, ".pose_raw.bin");
        dump(out.local_points, ".local_points.bin");
        if (out.depth.empty() && !out.local_points.empty()) {
            const size_t n = out.local_points.size() / 3;
            out.depth.resize(n);
            for (size_t i = 0; i < n; i++)
                out.depth[i] = out.local_points[3 * i + 2];
        }
        dump(out.depth, ".depth.bin");
        dump(out.conf, ".conf.bin");
        dump(out.world_points, ".points.bin");
        if (!out.mask.empty()) dump(out.mask, ".mask.bin");
        if (out.metric_scale > 0.0f) {
            std::vector<float> sc{out.metric_scale};
            dump(sc, ".scale.bin");
        }
    }

    const double t4 = steady_ms();

    // timing JSON: phase breakdown, mirrored to stderr (--timing) and the
    // meta.json sidecar so benchmark scripts can parse either. The last
    // run's per-phase values are kept for compatibility; aggregates over the
    // timed iterations describe steady-state latency.
    std::vector<std::string> tparts;
    auto tpush = [&tparts](const std::string& k, double v) {
        char buf[96];
        snprintf(buf, sizeof(buf), "\"%s\":%.2f", k.c_str(), v);
        tparts.push_back(buf);
    };
    auto tpush_raw = [&tparts](const std::string& k, const std::string& v) {
        tparts.push_back("\"" + k + "\":" + v);
    };
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2 - (v.size() % 2 == 0 ? 1 : 0)];
    };
    auto pctl = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0
                         : v[std::min((size_t)(p * (v.size() - 1) + 0.5),
                                      v.size() - 1)];
    };
    tpush("load_model", t1 - t0);
    tpush("preprocess", t2 - t1);
    for (const auto& [k, v] : out.timing_ms) tpush(k, v);
    tpush("write_outputs", t4 - t3);
    tpush("end_to_end", t4 - t0);
    if (repeats > 1 || warmup > 0) {
        tpush("warmup", warmup);
        tpush("repeats", repeats);
        tpush("inference_p50", med(iter_ms));
        tpush("inference_p95", pctl(iter_ms, 0.95));
        tpush("inference_mean", std::accumulate(iter_ms.begin(),
                                                iter_ms.end(), 0.0) /
                                   std::max(1, (int)iter_ms.size()));
        if (!iter_s1.empty()) tpush("stage1_p50", med(iter_s1));
        if (!iter_s2.empty()) tpush("stage2_p50", med(iter_s2));
        std::string arr = "[";
        for (size_t i = 0; i < iter_ms.size(); i++) {
            if (i) arr += ",";
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", iter_ms[i]);
            arr += buf;
        }
        tpush_raw("iterations_ms", arr + "]");
    }
    std::string tj;
    for (size_t i = 0; i < tparts.size(); i++) {
        if (i) tj += ",";
        tj += tparts[i];
    }
    if (timing) {
        fprintf(stderr,
                "{\"backend\":\"%s\",\"S\":%d,\"H\":%d,\"W\":%d,"
                "\"image_size\":%d,\"resize_mode\":\"%s\",\"timing_ms\":{%s}}\n",
                be.name.c_str(), S, H, W, image_size, resize_mode.c_str(),
                tj.c_str());
    }

    const std::string meta_path = out_prefix + ".meta.json";
    FILE* f = fopen(meta_path.c_str(), "w");
    if (f) {
        // dense outputs may live at a patch-aligned resolution that differs
        // from the input (original-VGGT at non-multiple sizes); report the
        // actual geometry when the runtime provides it
        const int oH = out.out_h > 0 ? out.out_h : H;
        const int oW = out.out_w > 0 ? out.out_w : W;
        const bool is_pi3 = !out.camera_poses.empty();
        const char* pose_layout = model->meta.architecture == "dust3r"
                                      ? "none"
                                      : (is_pi3 ? "S,16" : "S,9");
        fprintf(f,
                "{\"backend\":\"%s\",\"model\":\"%s\",\"S\":%d,\"H\":%d,"
                "\"W\":%d,\"image_size\":%d,"
                "\"layout\":{\"pose\":\"%s\",\"depth\":\"S,%d,%d\","
                "\"pts\":\"S,%d,%d,3\"},\"timing_ms\":{%s}}\n",
                be.name.c_str(), model_path.c_str(), S, H, W, image_size,
                pose_layout, oH, oW, oH, oW, tj.c_str());
        fclose(f);
    }
    return 0;
}

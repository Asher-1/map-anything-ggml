// C-API contract test for the mapggml integration surface.
//
//   test_capi <model.gguf>            full run (frames + image-view paths)
//   test_capi                         SKIP (exit 77)
//
// Checks: abi version, options NULL-safety, device enumeration, load-error
// last_error path, frames vs image-view input-path consistency (same source
// pixels must agree within u8 quantization), and multi-run stability
// (two consecutive runs bit-identical — the gallocr static-input regression).
#include "mapggml/capi.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* name, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                ok ? "" : " — ", detail.c_str());
    if (!ok) g_failures++;
}

// Deterministic LCG frames in [0,1], quantized to u8 so the image-view
// (u8) path and the frames (f32) path see IDENTICAL pixels.
float lcg_f32(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    const float q = (float)((state >> 8) & 0xFF) / 255.0f;
    return q;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "";
    if (!*model_path) {
        std::printf("SKIP: no GGUF model\nusage: test_capi <model.gguf>\n");
        return 77;
    }

    // ---- abi / options / devices ----
    check(mapggml_abi_version() == MAPGGML_CAPI_ABI_VERSION, "abi_version");
    mapggml_options_set_device(nullptr, "cpu");  // must be a no-op
    mapggml_options_free(nullptr);               // must be a no-op
    check(true, "options NULL-safety");

    check(mapggml_device_count() >= 2, "device_count >= auto+cpu",
          std::to_string(mapggml_device_count()));
    check(mapggml_device_at(0) != nullptr &&
              std::string(mapggml_device_at(0)->id) == "auto",
          "device_at(0) is auto");
    check(mapggml_device_at(mapggml_device_count()) == nullptr,
          "device_at out-of-range NULL");
    check(mapggml_device_available("cpu") == 1, "cpu available");
    check(mapggml_auto_device_order() != nullptr, "auto order string");

    // ---- load error path ----
    void* bad = mapggml_load("/nonexistent/model.gguf", nullptr);
    check(bad == nullptr, "load failure returns NULL");
    check(mapggml_last_error()[0] != '\0', "last_error populated on failure",
          mapggml_last_error());

    // ---- real load ----
    mapggml_options* opts = mapggml_options_new();
    mapggml_options_set_device(opts, "auto");
    void* ctx = mapggml_load(model_path, opts);
    mapggml_options_free(opts);
    if (!ctx) {
        check(false, "load", mapggml_last_error());
        return 1;
    }
    check(true, "load");

    const int S = 2;
    // pick the checkpoint's nominal resolution (patch-aligned for all six)
    int ISZ = mapggml_default_image_size(ctx);
    if (ISZ <= 0) ISZ = 512;
    const int H = ISZ, W = ISZ;
    std::vector<float> frames((size_t)S * 3 * H * W);
    std::vector<unsigned char> rgb8((size_t)H * W * 3);
    {
        unsigned int st = 7u;
        size_t i = 0;
        for (int s = 0; s < S; s++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++) {
                        const float v = lcg_f32(st);
                        frames[i++] = v;
                        if (s == 0 && c == 0)
                            rgb8[(size_t)y * W + x] =
                                (unsigned char)(v * 255.0f + 0.5f);
                    }
    }

    // ---- frames path, twice (multi-run stability) ----
    mapggml_result r1, r2;
    std::memset(&r1, 0, sizeof(r1));
    std::memset(&r2, 0, sizeof(r2));
    int rc = mapggml_run_frames(ctx, frames.data(), S, H, W, &r1);
    check(rc == 0, "run_frames", rc != 0 ? mapggml_last_error() : "");
    rc = mapggml_run_frames(ctx, frames.data(), S, H, W, &r2);
    check(rc == 0, "run_frames (second)", rc != 0 ? mapggml_last_error() : "");
    // dust3r carries no pose (pose_stride 0, NULL) — compare what exists
    const bool pose_ok = r1.pose == nullptr && r2.pose == nullptr;
    const bool pose_same =
        pose_ok || (r1.pose && r2.pose &&
                    r1.pose_stride == r2.pose_stride &&
                    std::memcmp(r1.pose, r2.pose,
                                (size_t)S * r1.pose_stride * sizeof(float)) ==
                        0);
    const bool depth_same =
        std::memcmp(r1.depth, r2.depth,
                    (size_t)S * H * W * sizeof(float)) == 0;
    check(pose_same && depth_same,
          "multi-run bit-identical (static-input regression)");

    // ---- image-view path: the SAME pixels as the frames path, re-encoded
    // through the u8 funnel (RGB8) for both views. Global attention makes
    // every view's output depend on every input, so BOTH views must match
    // the frames-path inputs for the comparison to be meaningful. ----
    std::vector<unsigned char> rgb8_all((size_t)S * H * W * 3);
    // re-pack (S,3,H,W) CHW frames into per-view HWC RGB8 (the image-view
    // memory convention) so both paths see identical pixels
    for (int v = 0; v < S; v++)
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    rgb8_all[(((size_t)v * H + y) * W + x) * 3 + c] =
                        (unsigned char)(
                            frames[((size_t)v * 3 + c) * H * W +
                                   (size_t)y * W + x] *
                                255.0f +
                            0.5f);
    mapggml_image_view views[2];
    for (int v = 0; v < 2; v++) {
        std::memset(&views[v], 0, sizeof(views[v]));
        views[v].data = rgb8_all.data() + (size_t)v * H * W * 3;
        views[v].width = W;
        views[v].height = H;
        views[v].row_stride_bytes = (size_t)W * 3;
        views[v].format = MAPGGML_IMAGE_RGB8;
    }

    mapggml_result r3;
    std::memset(&r3, 0, sizeof(r3));
    rc = mapggml_run_images(ctx, views, 2, &r3);
    check(rc == 0, "run_images", rc != 0 ? mapggml_last_error() : "");
    if (rc == 0 && r1.pose && r3.pose) {
        // The image-view path re-encodes the SAME source pixels through the
        // u8 funnel and the balanced resize; at the nominal (patch-aligned)
        // resolution the geometry is identical, so the depth fields must
        // agree within the u8 quantization + resize rounding margin.
        const bool same_geom =
            r3.height == r1.height && r3.width == r1.width;
        if (same_geom) {
            double max_d = 0.0, sum_d = 0.0;
            const size_t n = (size_t)r1.height * r1.width;
            for (size_t i = 0; i < n; i++) {
                const double d =
                    std::fabs((double)r1.depth[i] - r3.depth[i]);
                if (d > max_d) max_d = d;
                sum_d += d;
            }
            std::printf("  [diag] depth diff: max=%.6f mean=%.6f (n=%zu)\n",
                        max_d, sum_d / n, n);
            // the two paths receive bit-identical pixels (u8-exact source,
            // identity same-size resize) — require bit equality
            check(max_d == 0.0,
                  "image-view vs frames consistency (depth)",
                  "max_abs=" + std::to_string(max_d));
        } else {
            check(false, "image-view geometry matches frames path",
                  "r3 " + std::to_string(r3.height) + "x" +
                      std::to_string(r3.width));
        }
    }

    // ---- result sanity ----
    check(r1.architecture != nullptr && r1.architecture[0] != '\0',
          "architecture string", r1.architecture ? r1.architecture : "NULL");
    const std::string arch = r1.architecture ? r1.architecture : "";
    const bool no_pose_family = arch == "dust3r";
    check(no_pose_family ? (r1.pose == nullptr && r1.pose_stride == 0)
                         : (r1.pose != nullptr && r1.pose_stride > 0),
          "pose matches the family contract",
          "arch=" + arch);
    check(r1.depth != nullptr, "depth present");
    const bool finite_depth = [&] {
        for (size_t i = 0; i < (size_t)S * H * W; i++)
            if (!std::isfinite(r1.depth[i])) return false;
        return true;
    }();
    check(finite_depth, "depth finite (multi-run stability sanity)");

    // ---- format conversion + stride handling (BGR8 must agree with the
    // manually swapped RGB8 view bit-for-bit; an over-sized row_stride must
    // be honored) ----
    if (rc == 0) {
        // solid-color views: R=200 G=100 B=50 expressed in both orders
        const unsigned char px_rgb[3] = {200, 100, 50};
        const unsigned char px_bgr[3] = {50, 100, 200};
        // full-height buffers: rows are written per-stride below
        std::vector<unsigned char> rgb_buf((size_t)H * W * 3);
        const size_t vb_row_stride = (size_t)W * 3 + 7;  // over-sized stride
        std::vector<unsigned char> bgr_buf((size_t)H * vb_row_stride);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                for (int c = 0; c < 3; c++) {
                    rgb_buf[((size_t)y * W + x) * 3 + c] = px_rgb[c];
                    bgr_buf[(size_t)y * vb_row_stride + (size_t)x * 3 + c] =
                        px_bgr[c];
                }
            }
        }
        mapggml_image_view vr, vb;
        std::memset(&vr, 0, sizeof(vr));
        vr.data = rgb_buf.data();
        vr.width = W;
        vr.height = H;
        vr.row_stride_bytes = (size_t)W * 3;
        vr.format = MAPGGML_IMAGE_RGB8;
        std::memset(&vb, 0, sizeof(vb));
        vb.data = bgr_buf.data();
        vb.width = W;
        vb.height = H;
        vb.row_stride_bytes = vb_row_stride;  // stride padding honored
        vb.format = MAPGGML_IMAGE_BGR8;

        mapggml_result rr, rb;
        std::memset(&rr, 0, sizeof(rr));
        std::memset(&rb, 0, sizeof(rb));
        // dust3r is pair-wise: feed each format as a two-view pair
        mapggml_image_view pair_rgb[2] = {vr, vr};
        mapggml_image_view pair_bgr[2] = {vb, vb};
        int rc2 = mapggml_run_images(ctx, pair_rgb, 2, &rr);
        int rc3 = mapggml_run_images(ctx, pair_bgr, 2, &rb);
        check(rc2 == 0 && rc3 == 0, "format-branch runs",
              rc2 ? mapggml_last_error() : (rc3 ? mapggml_last_error() : ""));
        if (rc2 == 0 && rc3 == 0 && rr.depth && rb.depth &&
            rr.height == rb.height && rr.width == rb.width) {
            // self-consistency first: two runs of the SAME view must agree
            mapggml_result rr2;
            std::memset(&rr2, 0, sizeof(rr2));
            int rc4 = mapggml_run_images(ctx, pair_rgb, 2, &rr2);
            if (rc4 == 0 && rr2.depth) {
                const size_t n = (size_t)rr.height * rr.width;
                bool self_same =
                    std::memcmp(rr.depth, rr2.depth, n * sizeof(float)) == 0;
                std::printf("  [diag] RGB self-run identical: %s\n",
                            self_same ? "yes" : "NO");
                std::printf("  [diag] rgb depth[0..2] = %.4f %.4f %.4f\n",
                            rr.depth[0], rr.depth[1], rr.depth[2]);
                std::printf("  [diag] bgr depth[0..2] = %.4f %.4f %.4f\n",
                            rb.depth[0], rb.depth[1], rb.depth[2]);
            }
            mapggml_result_free(&rr2);
            mapggml_result rb2;
            std::memset(&rb2, 0, sizeof(rb2));
            int rc5 = mapggml_run_images(ctx, pair_bgr, 2, &rb2);
            if (rc5 == 0 && rb2.depth) {
                const size_t n = (size_t)rr.height * rr.width;
                bool bgr_repeat =
                    std::memcmp(rb.depth, rb2.depth, n * sizeof(float)) == 0;
                std::printf(
                    "  [diag] BGR repeat identical: %s | rb2[0..2] = "
                    "%.4f %.4f %.4f\n",
                    bgr_repeat ? "yes" : "NO", rb2.depth[0], rb2.depth[1],
                    rb2.depth[2]);
            }
            mapggml_result_free(&rb2);
            const size_t n = (size_t)rr.height * rr.width;
            check(std::memcmp(rr.depth, rb.depth,
                              n * sizeof(float)) == 0,
                  "BGR8 (+padded stride) == RGB8 bit-identical");
        }
        mapggml_result_free(&rr);
        mapggml_result_free(&rb);
    }

    mapggml_result_free(&r1);
    mapggml_result_free(&r2);
    mapggml_result_free(&r3);
    mapggml_result_free(nullptr);  // no-op
    mapggml_ctx_free(ctx);
    mapggml_free_buffer(nullptr);  // no-op

    if (g_failures) {
        std::printf("%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("ALL C-API TESTS PASSED\n");
    return 0;
}

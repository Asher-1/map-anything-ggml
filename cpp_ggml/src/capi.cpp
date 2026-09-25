// mapggml C ABI implementation. See include/mapggml/capi.h for the contract.
// Thin, exception-free translation layer over the C++ core (gguf_loader +
// VGGTRuntime + image_io); all host-facing buffers are malloc'd so the
// caller releases them with mapggml_result_free / mapggml_free_buffer.
#include "mapggml/capi.h"

#include "backend.hpp"
#include "common.hpp"
#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "image_io.hpp"
#include "vggt_graph.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace mapggml;

namespace {

thread_local std::string t_last_error;

void set_last_error(const char* msg) { t_last_error = msg ? msg : ""; }

double steady_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// C++-side snapshot of the opaque options handle.
struct Options {
    std::string device = "auto";
    int threads = 0;
    int image_size = 0;  // 0 = checkpoint's nominal resolution
    std::string resize_mode = "balanced";
};

// Session: one GGUF checkpoint + a lazily (re)built runtime for the current
// (S, H, W). A resolution change rebuilds the runtime (weights re-stage) —
// callers that sweep resolutions should expect that cost; keeping the input
// size across calls reuses everything.
struct Ctx {
    std::unique_ptr<GGUFModel> model;
    Backend backend;
    std::unique_ptr<VGGTRuntime> runtime;
    int cur_s = 0, cur_h = 0, cur_w = 0;
    Options opts;

    bool ensure_runtime(int s, int h, int w) {
        if (runtime && cur_s == s && cur_h == h && cur_w == w) return true;
        runtime.reset();
        cur_s = cur_h = cur_w = 0;
        runtime.reset(new VGGTRuntime(*model, backend, s, h, w));
        if (!runtime->valid()) {
            runtime.reset();
            return false;
        }
        cur_s = s;
        cur_h = h;
        cur_w = w;
        return true;
    }
};

// Fill `out` from a successful VGGTOutputs. All host-facing buffers are
// malloc'd copies (the C++ vectors die with the caller's stack frame).
void assemble_result(const GGUFModel& model, const VGGTOutputs& o, int S,
                     int H, int W, double preprocess_ms, double inference_ms,
                     double e2e_ms, mapggml_result* out) {
    auto dup = [](const std::vector<float>& v) -> float* {
        if (v.empty()) return nullptr;
        float* p = (float*)malloc(v.size() * sizeof(float));
        if (p) memcpy(p, v.data(), v.size() * sizeof(float));
        return p;
    };

    out->n_views = S;
    out->height = o.out_h > 0 ? o.out_h : H;
    out->width = o.out_w > 0 ? o.out_w : W;
    out->architecture = model.meta.architecture.c_str();

    const std::string& arch = model.meta.architecture;
    if (arch == "pi3" || arch == "pi3x" || arch == "mapanything") {
        out->pose = dup(o.camera_poses);
        out->pose_stride = 16;
    } else if (arch == "dust3r") {
        out->pose = nullptr;
        out->pose_stride = 0;
    } else {  // vggt_omega / vggt
        out->pose = dup(o.pose_enc);
        out->pose_stride = 9;
    }
    out->depth = dup(o.depth);
    // vggt family depth_conf vs pi3/dust3r conf — both land in `conf` with
    // the semantics documented in capi.h
    out->conf = dup(o.depth_conf.empty() ? o.conf : o.depth_conf);
    out->local_points = dup(o.local_points);
    out->points = dup(o.world_points);
    out->mask = dup(o.mask);
    if (o.metric_scale > 0.0f) {
        float* p = (float*)malloc(sizeof(float));
        if (p) *p = o.metric_scale;
        out->scale = p;
    } else {
        out->scale = nullptr;
    }
    out->text_embedding = dup(o.text_embedding);
    out->text_embedding_dim = (int32_t)o.text_embedding.size();

    out->timings.preprocess_ms = preprocess_ms;
    out->timings.inference_ms = inference_ms;
    out->timings.postprocess_ms = std::max(0.0, e2e_ms - preprocess_ms -
                                                    inference_ms);
    out->timings.e2e_ms = e2e_ms;
}

int run_common(Ctx* c, std::vector<float>& frames, int S, int H, int W,
               double preprocess_ms, double t_entry, mapggml_result* out) {
    if (!out) {
        set_last_error("result pointer is NULL");
        return -1;
    }
    if (!c->ensure_runtime(S, H, W)) {
        set_last_error("runtime build failed for the requested geometry");
        return -1;
    }
    VGGTOutputs o;
    const double t_inf0 = steady_ms();
    if (!c->runtime->run(frames.data(), o)) {
        set_last_error("inference failed");
        return -1;
    }
    const double t_inf1 = steady_ms();
    std::memset(out, 0, sizeof(*out));
    assemble_result(*c->model, o, S, H, W, preprocess_ms,
                    t_inf1 - t_inf0, steady_ms() - t_entry, out);
    t_last_error.clear();
    return 0;
}

}  // namespace

// ---- lifecycle ----

int mapggml_abi_version(void) { return MAPGGML_CAPI_ABI_VERSION; }

mapggml_options* mapggml_options_new(void) {
    try {
        return (mapggml_options*)(new Options());
    } catch (...) {
        return nullptr;
    }
}

void mapggml_options_free(mapggml_options* opts) {
    delete (Options*)opts;
}

void mapggml_options_set_device(mapggml_options* opts, const char* device) {
    if (!opts || !device || !*device) return;
    ((Options*)opts)->device = device;
}

void mapggml_options_set_threads(mapggml_options* opts, int n_threads) {
    if (!opts || n_threads < 0) return;
    ((Options*)opts)->threads = n_threads;
}

void mapggml_options_set_image_size(mapggml_options* opts, int image_size) {
    if (!opts || image_size < 0) return;
    ((Options*)opts)->image_size = image_size;  // 0 = model default
}

void mapggml_options_set_resize_mode(mapggml_options* opts, const char* mode) {
    if (!opts || !mode || !*mode) return;
    if (std::string(mode) != "balanced" && std::string(mode) != "max_size")
        return;
    ((Options*)opts)->resize_mode = mode;
}

void* mapggml_load(const char* gguf_path, const mapggml_options* opts) {
    if (!gguf_path || !*gguf_path) {
        set_last_error("gguf_path is NULL or empty");
        return nullptr;
    }
    try {
        Ctx* c = new Ctx();
        if (opts) c->opts = *(const Options*)opts;
        c->backend = init_backend(c->opts.device.c_str(), c->opts.threads);
        c->model = load_gguf(gguf_path);
        if (!c->model) {
            set_last_error("failed to load GGUF (see the process log)");
            delete c;
            return nullptr;
        }
        // early architecture check: create_builder logs the registered keys
        // when the GGUF's general.architecture is unknown
        auto probe = create_builder(c->model->meta.architecture);
        if (!probe) {
            set_last_error(("unknown architecture '" +
                            c->model->meta.architecture + "'")
                               .c_str());
            delete c;
            return nullptr;
        }
        t_last_error.clear();
        return c;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("unknown failure during load");
        return nullptr;
    }
}

void mapggml_ctx_free(void* ctx) { delete (Ctx*)ctx; }

// ---- inference ----

int mapggml_run_frames(void* ctx, const float* frames_f32, int n_views,
                       int height, int width, mapggml_result* out) {
    Ctx* c = (Ctx*)ctx;
    if (!c || !frames_f32 || n_views <= 0 || height <= 0 || width <= 0) {
        set_last_error("run_frames: invalid arguments");
        return -1;
    }
    try {
        const double t0 = steady_ms();
        std::vector<float> frames(
                frames_f32, frames_f32 + (size_t)n_views * 3 * height * width);
        return run_common(c, frames, n_views, height, width,
                          steady_ms() - t0, t0, out);
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("unknown failure in run_frames");
        return -1;
    }
}

int mapggml_run_images(void* ctx, const mapggml_image_view* views,
                       int n_views, mapggml_result* out) {
    Ctx* c = (Ctx*)ctx;
    if (!c || !views || n_views <= 0) {
        set_last_error("run_images: invalid arguments");
        return -1;
    }
    try {
        const double t0 = steady_ms();
        std::vector<float> frames;
        int H = 0, W = 0;
        const int patch = c->model->meta.patch_size;
        int image_size = c->opts.image_size;
        if (image_size <= 0)
            image_size = mapggml_default_image_size(c);
        if (image_size % patch != 0) {
            // keep the official invariant: resolution divisible by patch
            image_size = (image_size / patch) * patch;
            if (image_size < patch) image_size = patch;
        }
        if (!load_images_official_views(views, n_views, image_size, patch,
                                        frames, H, W,
                                        c->opts.resize_mode)) {
            set_last_error("image preprocessing failed");
            return -1;
        }
        {  // DEBUG: solid-color check — first 4 values of each channel plane
            const size_t HW = (size_t)H * W;
            char buf[220];
            std::snprintf(
                buf, sizeof(buf),
                "DEBUG y0x0=%.4f y1x0=%.4f y100x0=%.4f y517x0=%.4f "
                "c1y0x0=%.4f c2y0x0=%.4f",
                frames[0], frames[HW + W], frames[(size_t)100 * W],
                frames[(size_t)517 * W], frames[HW], frames[2 * HW]);
            log_info("%s", buf);
        }
        return run_common(c, frames, n_views, H, W, steady_ms() - t0, t0,
                          out);
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("unknown failure in run_images");
        return -1;
    }
}

int mapggml_patch_size(void* ctx) {
    Ctx* c = (Ctx*)ctx;
    return c && c->model ? c->model->meta.patch_size : 0;
}

int mapggml_default_image_size(void* ctx) {
    Ctx* c = (Ctx*)ctx;
    if (!c || !c->model) return 0;
    const std::string& arch = c->model->meta.architecture;
    // dust3r ships a 512_dpt checkpoint; the omega 512/416 variants name
    // their nominal resolution; everything else gates at 518
    if (arch == "dust3r") return 512;
    if (arch == "vggt_omega") return 512;
    return 518;
}

void mapggml_result_free(mapggml_result* r) {
    if (!r) return;
    free((void*)r->pose);
    free((void*)r->depth);
    free((void*)r->conf);
    free((void*)r->local_points);
    free((void*)r->points);
    free((void*)r->mask);
    free((void*)r->scale);
    free((void*)r->text_embedding);
    std::memset(r, 0, sizeof(*r));
}

void mapggml_free_buffer(void* p) { free(p); }

const char* mapggml_last_error(void) { return t_last_error.c_str(); }

// ---- device enumeration (delegates to the backend layer) ----

int mapggml_device_count(void) { return backend_device_count(); }

const mapggml_device_info* mapggml_device_at(int index) {
    return (const mapggml_device_info*)backend_device_at(index);
}

const char* mapggml_auto_device_order(void) {
    return backend_auto_device_order();
}

int mapggml_device_available(const char* device) {
    return backend_device_available(device);
}

int mapggml_warmup_backend(const char* device) {
    try {
        Backend be = init_backend(device ? device : "auto", 0);
        return be.handle ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

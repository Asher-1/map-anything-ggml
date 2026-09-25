// ----------------------------------------------------------------------------
// mapggml C ABI — integration surface for host applications (ACloudViewer
// AICore, plugins, Python bindings). Mirrors the AICore C-API contract:
// opaque contexts, explicit options, typed results with release functions,
// thread-local last_error, borrowed stride-aware image views, and NO
// environment variables for logic control. No STL, Qt, OpenCV, exceptions,
// or ggml types cross this boundary.
// ----------------------------------------------------------------------------
#ifndef MAPGGML_CAPI_H
#define MAPGGML_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MAPGGML_SHARED)
#ifdef MAPGGML_BUILD
#define MAPGGML_CAPI __declspec(dllexport)
#else
#define MAPGGML_CAPI __declspec(dllimport)
#endif
#elif defined(MAPGGML_BUILD)
#define MAPGGML_CAPI __attribute__((visibility("default")))
#else
#define MAPGGML_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Binary contract of this header. The implementation returns it; bump when
 * any struct layout or function signature changes. */
#define MAPGGML_CAPI_ABI_VERSION 1
MAPGGML_CAPI int mapggml_abi_version(void);

/* ---- borrowed image view (layout-compatible with AICore's
 * aicore_image_view: a host adapter can convert field by field) ---- */

typedef enum mapggml_image_format {
    MAPGGML_IMAGE_RGB8 = 1,
    MAPGGML_IMAGE_RGBA8 = 2,
    MAPGGML_IMAGE_GRAY8 = 3,
    MAPGGML_IMAGE_BGR8 = 4,
    MAPGGML_IMAGE_BGRA8 = 5
} mapggml_image_format;

typedef struct mapggml_image_view {
    const uint8_t* data;        /* borrowed, read-only, alive for the call */
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;    /* bytes between consecutive rows */
    mapggml_image_format format;
} mapggml_image_view;

/* ---- options (opaque, heap-allocated, value semantics; all setters are
 * no-ops on NULL and ignore invalid values, keeping the previous state) ---- */

typedef struct mapggml_options mapggml_options;

MAPGGML_CAPI mapggml_options* mapggml_options_new(void);
MAPGGML_CAPI void mapggml_options_free(mapggml_options* opts);

/* "auto|cpu|cuda|vulkan|metal" or an instance name such as "CUDA0".
 * NULL/empty keeps the current value. "auto" picks the first available
 * compiled-in backend (CUDA -> Metal -> Vulkan -> CPU). */
MAPGGML_CAPI void mapggml_options_set_device(mapggml_options* opts,
                                             const char* device);
/* 0 = backend default thread count (CPU backend only). */
MAPGGML_CAPI void mapggml_options_set_threads(mapggml_options* opts,
                                              int n_threads);
/* Preprocess longest-side target before inference. 0 (the default) uses
 * the checkpoint's nominal resolution (mapggml_default_image_size);
 * non-zero values are clamped to a patch multiple. Matches the official
 * demo image_resolution. */
MAPGGML_CAPI void mapggml_options_set_image_size(mapggml_options* opts,
                                                 int image_size);
/* "balanced" (official default) or "max_size". NULL/empty keeps the current
 * value. Only affects the image-view entry points. */
MAPGGML_CAPI void mapggml_options_set_resize_mode(mapggml_options* opts,
                                                  const char* mode);

/* ---- context lifecycle ---- */

/* Load a GGUF checkpoint. The architecture (vggt_omega / vggt / pi3 / pi3x /
 * mapanything / dust3r) is detected from the GGUF and dispatched internally:
 * ONE context type serves all six models. The checkpoint's mean/std travels
 * in the GGUF and is applied as the first graph op. NULL on failure (see
 * mapggml_last_error). opts may be NULL (all defaults). */
MAPGGML_CAPI void* mapggml_load(const char* gguf_path,
                                const mapggml_options* opts);
/* Contexts are not thread-safe: serialize run calls (one worker at a time),
 * distinct contexts may run concurrently. */
MAPGGML_CAPI void mapggml_ctx_free(void* ctx);

/* ---- inference ---- */

/* Common wall-clock timings (semantics follow AICore's pipeline timings:
 * preprocess = decode/resize/normalize + graph-input preparation;
 * inference = input upload + graph execution + readback; postprocess =
 * typed-result construction; e2e = entry until the result is ready). */
typedef struct mapggml_timings {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double e2e_ms;
} mapggml_timings;

/* Typed result for one multi-view inference. Buffer ownership: every
 * non-const pointer was malloc'd by the library and is released by
 * mapggml_result_free (safe on NULL). Which fields are populated is an
 * architecture property — see the per-family notes below. */
typedef struct mapggml_result {
    int32_t n_views;
    int32_t height;         /* processed geometry (dense outputs) */
    int32_t width;
    const char* architecture; /* "vggt_omega" "vggt" "pi3" "pi3x"
                                 "mapanything" "dust3r"; owned by the
                                 context — valid until mapggml_ctx_free */
    mapggml_timings timings;

    /* Camera outputs — three families:
     *   vggt_omega / vggt : pose_stride = 9, pose = pose_enc [n*9]
     *                       (absT 3 + quaternion 4 + FoV 2, FoV = relu+0.01)
     *   pi3 / pi3x / mapanything : pose_stride = 16, pose = camera-to-world
     *                       4x4 row-major [n*16] (translation in flat
     *                       3/7/11; pi3 family pose is orthogonalized
     *                       host-side from the raw head output)
     *   dust3r : pose_stride = 0, pose = NULL (no pose head; the official
     *                       poses come from an out-of-network optimizer) */
    const float* pose;
    int32_t pose_stride;      /* floats per view: 9 / 16 / 0 */

    /* Dense depth [n*H*W] view-major. vggt family: the dedicated depth head
     * (exp of logits). pi3 family: camera-frame z of the local points (the
     * official convention). dust3r: camera-frame z of the pointmaps. */
    const float* depth;

    /* Confidence. vggt family: depth_conf [n*H*W] = 1+exp(logits).
     * pi3 family: conf [n*H*W] holds RAW logits (sigmoid for probabilities).
     * dust3r: conf [n*H*W] = 1+exp. NULL when the model has none. */
    const float* conf;

    /* Per-view local pointmaps [n*H*W*3] (camera frame, xyz).
     * pi3 / pi3x / mapanything / dust3r only (NULL for vggt family).
     * dust3r: BOTH views live in view1's frame (s0 = head1 self-view,
     * s1 = head2 other-view). */
    const float* local_points;

    /* World/common-frame pointmaps [n*H*W*3]. vggt-1b: the official point
     * head (view0 frame). mapanything: the common-frame points. dust3r:
     * view2's pointmap in view1's frame (companion of local_points[1]).
     * NULL when the model has no point head. */
    const float* points;

    /* mapanything only: non-ambiguous mask [n*H*W] (sigmoid probability). */
    const float* mask;
    /* mapanything only: metric scale, exp of the scale-token head ([1]). */
    const float* scale;
    /* vggt-omega text-alignment variant only: L2-normalized language-aligned
     * embedding [2048]. Empty (NULL, 0) otherwise. */
    const float* text_embedding;
    int32_t text_embedding_dim;
} mapggml_result;

/* Multi-view inference from DECODED images (borrowed views; the official
 * balanced/max_size preprocessing runs inside, including white padding of
 * mixed sizes). n_views >= 1; every view must be non-NULL with positive
 * dimensions. Returns 0 ok, -1 error (mapggml_last_error). */
MAPGGML_CAPI int mapggml_run_images(void* ctx,
                                    const mapggml_image_view* views,
                                    int n_views,
                                    mapggml_result* out);

/* Multi-view inference from preprocessed frames — the bit-exact parity path
 * (same contract as the CLI --bin feed): frames_f32 is n*3*H*W floats in
 * [0,1], torch (n,3,H,W) order. The result's height/width echo H/W. */
MAPGGML_CAPI int mapggml_run_frames(void* ctx, const float* frames_f32,
                                    int n_views, int height, int width,
                                    mapggml_result* out);

/* Model metadata needed by hosts to pick a sane input size: patch size
 * (14 for the vggt/pi3 families, 16 for dust3r) and the checkpoint's
 * default preprocessing resolution (512 for the vggt-omega variants and
 * dust3r, 518 for the others). */
MAPGGML_CAPI int mapggml_patch_size(void* ctx);
MAPGGML_CAPI int mapggml_default_image_size(void* ctx);

/* Releases buffers owned by a result. Safe on NULL. */
MAPGGML_CAPI void mapggml_result_free(mapggml_result* r);
/* Releases any malloc'd buffer handed out by this API. Safe on NULL. */
MAPGGML_CAPI void mapggml_free_buffer(void* p);

/* ---- diagnostics ---- */

/* Last error on the calling thread (empty string when none). The pointer
 * stays valid until the next mapggml call on that thread. */
MAPGGML_CAPI const char* mapggml_last_error(void);

/* ---- backend/device enumeration (for UI dropdowns) ---- */

typedef struct {
    const char* id;      /* device string accepted by options_set_device */
    const char* label;   /* human-readable */
    int is_default;      /* 1 for the "auto" entry */
} mapggml_device_info;

/* Number of discovered devices plus the synthetic "auto" entry (>= 1). */
MAPGGML_CAPI int mapggml_device_count(void);
/* Device info at 0-based index; NULL when out of range. Pointers are valid
 * for the lifetime of the process. */
MAPGGML_CAPI const mapggml_device_info* mapggml_device_at(int index);
/* Human-readable auto-pick order, e.g. "CUDA -> Vulkan -> CPU". */
MAPGGML_CAPI const char* mapggml_auto_device_order(void);
/* 1 when the requested device resolves to an available backend. */
MAPGGML_CAPI int mapggml_device_available(const char* device);
/* Lightweight backend registration + device verification; call on the UI
 * thread before spawning a worker. Returns 0 ok, -1 unavailable. */
MAPGGML_CAPI int mapggml_warmup_backend(const char* device);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MAPGGML_CAPI_H */

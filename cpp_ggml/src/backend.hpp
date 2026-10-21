// Backend selection for mapggml (six-model ggml inference).
// GPU backends are compiled in via MAP_USE_CUDA / MAP_USE_METAL /
// MAP_USE_VULKAN; CPU is always available. The device string ("auto", "cpu",
// "cuda", "vulkan", "metal", or an instance name such as "CUDA0") picks
// among the compiled-in backends at RUNTIME — no environment variables.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

namespace mapggml {

struct Backend {
    ggml_backend_t handle = nullptr;
    std::string name = "CPU";
};

// Explicit backend options — every field replaces a former process-global
// switch; this interface has no environment variables (AICore integration
// style: rmbg GraphOptions / aicore_*_options setters).
struct BackendOptions {
    // "auto|cpu|cuda|vulkan|metal" or an instance name such as "CUDA0".
    std::string device = "auto";
    // CPU thread count; <= 0 picks hardware_concurrency (CPU backend only).
    int cpu_threads = 0;
    // Vulkan f32-accuracy profile (was the GGML_VK_DISABLE_F16/COOPMAT/
    // COOPMAT2/DOT2/INTEGER_DOT_PRODUCT environment knobs): disables the
    // fp16/coop-mat shader paths whose kernels drift ~20x past the f16 CPU
    // noise floor through deep decoder stacks (querysplat gate 13/15 FAIL,
    // gaussians corr 0.968); with the profile the Vulkan gate equals the CPU
    // floor and still runs several-x faster than CPU.
    bool vulkan_f32_accuracy = true;
    // CUDA f32-accuracy profile (was the GGML_CUDA_DISABLE_TF32 environment
    // knob): selects exact-F32 cuBLAS GEMMs — ggml-cuda otherwise runs them
    // in TF32 tensor-core math (10-bit mantissa), while torch defaults to
    // allow_tf32 = false. The flash-attention F32->F16 K/V conversion is
    // bypassed separately at the graph level (RuntimeOptions::f32_accuracy).
    bool cuda_f32_accuracy = true;
};

// Initialize the compiled-in backend from explicit options. Returns a
// Backend whose handle is owned by ggml (do not free). The accuracy profiles
// are consumed ONCE at the first device init of each backend in the process,
// so init_backend applies them here — the options are the only configuration
// surface (vendored ggml transports the resolved values to its device-init
// reads internally; that mechanism is not part of this interface). "auto"
// picks the first available compiled-in GPU backend (CUDA -> Metal ->
// Vulkan) and falls back to CPU; an explicit device that is not compiled in
// or has no device falls back to CPU with a logged warning. Dies with a
// clear message when nothing is available.
Backend init_backend(const BackendOptions& opts);
// Deprecated convenience overloads (default accuracy profile).
Backend init_backend(const char* device, int cpu_threads = 0);
// Convenience overload: "auto".
Backend init_backend(int cpu_threads = 0);

// ---- device enumeration (for UI dropdowns / the C API) ----

struct DeviceEntry {
    std::string id;      // "auto" "cpu" "cuda" "vulkan" "metal" "CUDA0" ...
    std::string label;   // human-readable
    bool is_default = false;
};

// Discovered devices plus the synthetic "auto" entry (>= 1).
int backend_device_count();
const DeviceEntry* backend_device_at(int index);
// Human-readable auto-pick order, e.g. "CUDA -> Vulkan -> CPU" (static).
const char* backend_auto_device_order();
// 1 when the device string resolves to an available compiled-in backend.
int backend_device_available(const char* device);

}  // namespace mapggml

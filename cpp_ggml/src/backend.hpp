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

// Initialize the compiled-in backend. Returns a Backend whose handle is owned
// by ggml (do not free). "auto" picks the first available compiled-in GPU
// backend (CUDA -> Metal -> Vulkan) and falls back to CPU; an explicit device
// that is not compiled in or has no device falls back to CPU with a logged
// warning. Dies with a clear message when nothing is available.
Backend init_backend(const char* device, int cpu_threads = 0);
// Convenience overload: "auto".
Backend init_backend(int cpu_threads = 0);

// Opt the Vulkan backend into its f32-accuracy mode: ggml-vulkan picks its
// shader paths from environment knobs read ONCE at device init, and the
// default fp16/coop-mat kernels drift ~20x past the f16 CPU noise floor
// through deep decoder stacks (querysplat gate 13/15 FAIL, gaussians corr
// 0.968). With the knobs disabled the Vulkan gate equals the CPU floor and
// still runs several-x faster than CPU. Call BEFORE init_backend(); user-set
// variables are never overwritten. No-op when Vulkan is not compiled in.
void set_vulkan_f32_accuracy_mode(bool enable = true);

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

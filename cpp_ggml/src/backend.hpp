// Backend selection for mapggml (VGGT-Omega ggml inference).
// Mirrors the ultralytics-ggml / lingbot-map-ggml wiring: exactly one GPU
// backend is compiled in via MAP_USE_CUDA / MAP_USE_METAL / MAP_USE_VULKAN;
// CPU is always available as fallback.
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
// by ggml (do not free). Dies with a clear message when nothing is available.
Backend init_backend(int cpu_threads = 0);

}  // namespace mapggml

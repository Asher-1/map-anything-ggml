#include "backend.hpp"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <thread>

#ifdef MAP_USE_CUDA
#  include "ggml-cuda.h"
#endif
#ifdef MAP_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef MAP_USE_VULKAN
#  include "ggml-vulkan.h"
#endif

namespace mapggml {

Backend init_backend(int cpu_threads) {
    Backend b;
#ifdef MAP_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        b.handle = ggml_backend_cuda_init(0);
        if (b.handle) {
            b.name = "CUDA0";
            return b;
        }
    }
    fprintf(stderr, "MAP_GGML_CUDA build but no CUDA device; falling back to CPU\n");
#endif
#ifdef MAP_USE_METAL
    b.handle = ggml_backend_metal_init();
    if (b.handle) {
        b.name = "Metal";
        return b;
    }
#endif
#ifdef MAP_USE_VULKAN
    b.handle = ggml_backend_vk_init(0);
    if (b.handle) {
        b.name = "Vulkan0";
        return b;
    }
#endif
    (void)b;
    b.handle = ggml_backend_cpu_init();
    b.name = "CPU";
    // ggml's library default is GGML_DEFAULT_N_THREADS == 4, which leaves a
    // 32-core box ~2x slower than tuned (measured: pi3x q8_0 24.9s @4 vs
    // 13.1s @24).  Default to hardware_concurrency; --threads overrides.
    int n = cpu_threads;
    if (n <= 0)
        n = (int)std::thread::hardware_concurrency();
    if (n > 0)
        ggml_backend_cpu_set_n_threads(b.handle, n);
    return b;
}

}  // namespace mapggml

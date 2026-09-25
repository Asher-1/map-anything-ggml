#include "backend.hpp"
#include "common.hpp"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

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

namespace {

bool have_cuda() {
#ifdef MAP_USE_CUDA
    return ggml_backend_cuda_get_device_count() > 0;
#else
    return false;
#endif
}
bool have_metal() {
#ifdef MAP_USE_METAL
    return true;  // presence decided by ggml_backend_metal_init at init time
#else
    return false;
#endif
}
bool have_vulkan() {
#ifdef MAP_USE_VULKAN
    return true;
#else
    return false;
#endif
}

ggml_backend_t try_cuda() {
#ifdef MAP_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0)
        return ggml_backend_cuda_init(0);
#endif
    return nullptr;
}
ggml_backend_t try_metal() {
#ifdef MAP_USE_METAL
    return ggml_backend_metal_init();
#else
    return nullptr;
#endif
}
ggml_backend_t try_vulkan() {
#ifdef MAP_USE_VULKAN
    return ggml_backend_vk_init(0);
#else
    return nullptr;
#endif
}

void set_cpu_threads(ggml_backend_t handle, int cpu_threads) {
    // ggml's library default is GGML_DEFAULT_N_THREADS == 4, which leaves a
    // 32-core box ~2x slower than tuned (measured: pi3x q8_0 24.9s @4 vs
    // 13.1s @24). Default to hardware_concurrency; caller overrides.
    int n = cpu_threads;
    if (n <= 0) n = (int)std::thread::hardware_concurrency();
    if (n > 0) ggml_backend_cpu_set_n_threads(handle, n);
}

std::string to_lower_ascii(const char* s) {
    std::string out = s ? s : "";
    for (char& ch : out)
        if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
    return out;
}

}  // namespace

Backend init_backend(const char* device, int cpu_threads) {
    Backend b;
    const std::string want = to_lower_ascii(device);
    ggml_backend_t gpu = nullptr;

    if (want.empty() || want == "auto" || want == "gpu") {
        // auto pick order: CUDA -> Metal -> Vulkan -> CPU
        if ((gpu = try_cuda())) {
            b.handle = gpu;
            b.name = "CUDA0";
            return b;
        }
        if ((gpu = try_metal())) {
            b.handle = gpu;
            b.name = "Metal";
            return b;
        }
        if ((gpu = try_vulkan())) {
            b.handle = gpu;
            b.name = "Vulkan0";
            return b;
        }
    } else if (want == "cuda" || want.rfind("cuda", 0) == 0) {
        if ((gpu = try_cuda())) {
            b.handle = gpu;
            b.name = "CUDA0";
            return b;
        }
        log_warn("device '%s' requested but CUDA is unavailable in this "
                 "build; falling back to CPU",
                 device);
    } else if (want == "metal") {
        if ((gpu = try_metal())) {
            b.handle = gpu;
            b.name = "Metal";
            return b;
        }
        log_warn("device '%s' requested but Metal is unavailable in this "
                 "build; falling back to CPU",
                 device);
    } else if (want == "vulkan" || want.rfind("vulkan", 0) == 0) {
        if ((gpu = try_vulkan())) {
            b.handle = gpu;
            b.name = "Vulkan0";
            return b;
        }
        log_warn("device '%s' requested but Vulkan is unavailable in this "
                 "build; falling back to CPU",
                 device);
    } else if (want != "cpu") {
        log_warn("unknown device '%s'; falling back to CPU", device);
    }

    b.handle = ggml_backend_cpu_init();
    b.name = "CPU";
    if (!b.handle) {
        log_error("no backend could be initialized (device request: '%s')",
                  device ? device : "auto");
        return b;
    }
    set_cpu_threads(b.handle, cpu_threads);
    return b;
}

Backend init_backend(int cpu_threads) { return init_backend("auto", cpu_threads); }

// ---- device enumeration ----

int backend_device_count() {
    int n = 1;  // the synthetic "auto" entry
    if (have_cuda()) n++;
    if (have_metal()) n++;
    if (have_vulkan()) n++;
    n++;  // cpu
    return n;
}

const DeviceEntry* backend_device_at(int index) {
    static std::vector<DeviceEntry> table;
    if (table.empty()) {
        DeviceEntry auto_e;
        auto_e.id = "auto";
        auto_e.label = "Auto (" + std::string(backend_auto_device_order()) + ")";
        auto_e.is_default = true;
        table.push_back(auto_e);
        if (have_cuda())
            table.push_back({"cuda", "CUDA (NVIDIA GPU)", false});
        if (have_metal()) table.push_back({"metal", "Metal (Apple GPU)", false});
        if (have_vulkan())
            table.push_back({"vulkan", "Vulkan (GPU)", false});
        table.push_back({"cpu", "CPU", false});
    }
    if (index < 0 || index >= (int)table.size()) return nullptr;
    return &table[index];
}

const char* backend_auto_device_order() {
    static const std::string order = [] {
        std::string o;
        if (have_cuda()) o += "CUDA -> ";
        if (have_metal()) o += "Metal -> ";
        if (have_vulkan()) o += "Vulkan -> ";
        o += "CPU";
        return o;
    }();
    return order.c_str();
}

int backend_device_available(const char* device) {
    const std::string want = to_lower_ascii(device);
    if (want.empty() || want == "auto" || want == "gpu")
        return 1;  // always resolvable: CPU is the built-in fallback
    if (want == "cpu") return 1;
    if (want.rfind("cuda", 0) == 0) return have_cuda() ? 1 : 0;
    if (want == "metal") return have_metal() ? 1 : 0;
    if (want.rfind("vulkan", 0) == 0) return have_vulkan() ? 1 : 0;
    return 0;
}

}  // namespace mapggml

// Shared helpers for mapggml (VGGT-Omega ggml inference).
#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mapggml {

// A weight tensor staged on the host, straight out of the GGUF file.
struct HostTensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    std::vector<uint8_t> data;  // raw bytes in ggml layout
};

inline size_t nbytes_of(enum ggml_type type, const int64_t ne[4]) {
    const size_t tsz = ggml_type_size(type);
    const int64_t nelements = ne[0] * ne[1] * ne[2] * ne[3];
    // Block types pack GGML_BLCK_SIZE elements per block along ne0.
    const int bs = ggml_blck_size(type);
    return (size_t)((nelements / bs) * tsz);
}

// Simple stderr logging with a mapggml prefix.
void log_info(const char* fmt, ...);
void log_error(const char* fmt, ...);

// Load a whole binary file (raw f32 image buffers for the CLI).
std::vector<uint8_t> read_file_bytes(const std::string& path);

// Dump a ggml tensor's contents to a raw file (parity debugging).
bool dump_tensor(const struct ggml_tensor* t, const std::string& path);

}  // namespace mapggml

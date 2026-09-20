cmake_minimum_required(VERSION 3.14)
project(mapggml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options (mirrors the ultralytics-ggml / lingbot-map-ggml integration style)
option(MAP_GGML_BUILD_CLI   "Build the vggt-cli binary"             ON)
option(MAP_GGML_SHARED      "Build libmapggml as a shared library"  OFF)
option(MAP_GGML_BUILD_TESTS "Build the regression tests (need a GGUF to run)" ON)

# Backend forwarding to ggml. At most one GPU backend per build directory;
# CPU is always included (GPU builds fall back to CPU for unsupported ops).
option(MAP_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(MAP_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(MAP_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)
option(MAP_GGML_HIPBLAS "Enable HIPBLAS backend in ggml" OFF)

# Compile the parity stage-dump hooks (MAPGGML_DUMP_STAGE env at runtime).
# Production builds keep this OFF to save binary size and compile time.
option(MAPGGML_ENABLE_DUMP "Compile parity stage-dump hooks" OFF)

set(_map_gpu_backends 0)
foreach(_backend MAP_GGML_CUDA MAP_GGML_METAL MAP_GGML_VULKAN MAP_GGML_HIPBLAS)
    if(${_backend})
        math(EXPR _map_gpu_backends "${_map_gpu_backends} + 1")
    endif()
endforeach()
if(_map_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

# ggml uses CMake's `native` CUDA architecture only with CMake >=3.24. On
# older CMake, select the attached GPU instead of compiling a broad list that
# can include device intrinsics unavailable in the installed CUDA toolkit.
if(MAP_GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_LESS 3.24)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _map_cuda_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _map_cuda_compute_cap "${_map_cuda_compute_cap}")
        if(_map_cuda_compute_cap)
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_map_cuda_compute_cap}")
            message(STATUS "CUDA architecture auto-detected: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
endif()

if(MAP_GGML_CUDA)
    set(GGML_CUDA        ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Replay CUDA graphs" FORCE)
endif()
if(MAP_GGML_METAL)
    set(GGML_METAL   ON CACHE BOOL "" FORCE)
endif()
if(MAP_GGML_VULKAN)
    set(GGML_VULKAN  ON CACHE BOOL "" FORCE)
endif()
if(MAP_GGML_HIPBLAS)
    set(GGML_HIP     ON CACHE BOOL "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml.
#
# GGML_LLAMAFILE=ON enables ggml's tinyBLAS path (an AVX-512 / AVX2 tiled
# SGEMM for F32 x F32 mul_mat) — a big win for the F32 conv hotspots via the
# im2col+mul_mat pipeline, with graceful fallback on older hosts.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# GGML_LLAMAFILE=ON enables ggml's tinyBLAS path (an AVX-512 / AVX2 tiled
# SGEMM for F32 x F32 mul_mat) — a big win for the F32 conv hotspots via the
# im2col+mul_mat pipeline. `-DGGML_LLAMAFILE=OFF` overrides it (bisecting
# suspected tinyBLAS numeric issues).
option(GGML_LLAMAFILE "Enable tinyBLAS SGEMM kernels in ggml" ON)

# Apply the local ggml integration patches before we descend into the
# submodule. They live in third_party/ggml-patches/ and are applied
# idempotently; zero patches is a valid configuration. See
# scripts/apply_ggml_patches.sh; it can also be invoked standalone.
execute_process(
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh
    RESULT_VARIABLE _ggml_patch_result
    OUTPUT_VARIABLE _ggml_patch_output
    ERROR_VARIABLE  _ggml_patch_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT _ggml_patch_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to apply the ggml integration patch (exit ${_ggml_patch_result}):\n"
        "stdout:\n${_ggml_patch_output}\n"
        "stderr:\n${_ggml_patch_error}")
endif()
message(STATUS "${_ggml_patch_output}")

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# stb (header-only interface target)
add_library(map_stb INTERFACE)
target_include_directories(map_stb INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

set(MAPGGML_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/gguf_loader.cpp
    src/backend.cpp
    src/vggt_graph.cpp
)

if(MAP_GGML_SHARED)
    add_library(mapggml SHARED ${MAPGGML_SOURCES})
else()
    add_library(mapggml STATIC ${MAPGGML_SOURCES})
endif()

# Propagate the active GPU backend to src/backend.cpp so it knows which
# ggml_backend_*_init() to call. ggml itself is configured above via the
# GGML_* cache vars; these definitions just gate our wiring code.
if(MAP_GGML_CUDA)
    target_compile_definitions(mapggml PRIVATE MAP_USE_CUDA)
endif()
if(MAP_GGML_METAL)
    target_compile_definitions(mapggml PRIVATE MAP_USE_METAL)
endif()
if(MAP_GGML_VULKAN)
    target_compile_definitions(mapggml PRIVATE MAP_USE_VULKAN)
endif()
if(MAP_GGML_HIPBLAS)
    # HIP uses the CUDA backend code path in ggml (ggml_backend_cuda_init
    # works for HIP builds). Expose the same define.
    target_compile_definitions(mapggml PRIVATE MAP_USE_CUDA)
endif()
if(MAPGGML_ENABLE_DUMP)
    target_compile_definitions(mapggml PRIVATE MAPGGML_ENABLE_DUMP)
endif()

target_include_directories(mapggml
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mapggml
    PUBLIC  ggml
    PRIVATE map_stb)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mapggml PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mapggml PRIVATE MAP_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mapggml PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(MAP_GGML_BUILD_CLI)
    add_executable(vggt-cli src/cli.cpp)
    target_link_libraries(vggt-cli PRIVATE mapggml)
endif()

if(MAP_GGML_BUILD_TESTS)
    add_executable(test_s2_independence tests/test_s2_independence.cpp)
    target_link_libraries(test_s2_independence PRIVATE mapggml)
    # regression runner lives in tests/test_matrix.py (no build step needed)
endif()

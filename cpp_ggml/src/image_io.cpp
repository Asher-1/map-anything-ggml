#include "image_io.hpp"

#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mapggml {

namespace {

// Python round(): round half to even (the official loader relies on
// builtins.round / np.round for every target-shape decision)
int round_half_even(double v) {
    const double f = std::floor(v);
    const double diff = v - f;
    if (diff > 0.5) return (int)f + 1;
    if (diff < 0.5) return (int)f;
    return ((int)f % 2 == 0) ? (int)f : (int)f + 1;
}

struct ImageBuf {
    int w = 0, h = 0;
    std::vector<unsigned char> px;  // RGB uint8, row-major
};

// official _crop_to_supported_aspect_ratio: center-crop extreme aspect
// ratios (height/width) into [0.5, 2.0]
void crop_to_supported_aspect(ImageBuf& im) {
    const double aspect = (double)im.h / (double)std::max(im.w, 1);
    if (aspect < 0.5) {
        const int crop_w =
            std::min(im.w, std::max(1, round_half_even(im.h / 0.5)));
        const int left = std::max((im.w - crop_w) / 2, 0);
        ImageBuf out;
        out.w = crop_w;
        out.h = im.h;
        out.px.resize((size_t)crop_w * im.h * 3);
        for (int y = 0; y < im.h; y++) {
            memcpy(&out.px[(size_t)y * crop_w * 3],
                   &im.px[((size_t)y * im.w + left) * 3], (size_t)crop_w * 3);
        }
        im = std::move(out);
    } else if (aspect > 2.0) {
        const int crop_h =
            std::min(im.h, std::max(1, round_half_even(im.w * 2.0)));
        const int top = std::max((im.h - crop_h) / 2, 0);
        ImageBuf out;
        out.w = im.w;
        out.h = crop_h;
        out.px.resize((size_t)im.w * crop_h * 3);
        memcpy(out.px.data(), &im.px[(size_t)top * im.w * 3],
               (size_t)im.w * crop_h * 3);
        im = std::move(out);
    }
}

}  // namespace

// Official load_and_preprocess_images(mode="balanced") equivalent:
//   1) center-crop extreme aspect ratios into [0.5, 2.0]
//   2) balanced target shape: token budget (res/patch)^2 split by aspect
//      ratio, both sides rounded to patch multiples (round-half-even)
//   3) BICUBIC resize (PIL == Catmull-Rom, Keys a=-0.5)
//   4) pad differing shapes to the common max size with white (1.0)
bool load_images_official(const std::vector<std::string>& paths,
                          int image_resolution, int patch_size,
                          std::vector<float>& out, int& height, int& width,
                          const std::string& mode) {
    if (paths.empty()) {
        log_error("no input images");
        return false;
    }
    if (image_resolution <= 0 || image_resolution % patch_size != 0) {
        log_error("image_resolution (%d) must be positive and divisible by "
                  "patch_size (%d)",
                  image_resolution, patch_size);
        return false;
    }
    if (mode != "balanced" && mode != "max_size") {
        log_error("unknown resize mode: %s (balanced|max_size)", mode.c_str());
        return false;
    }
    out.clear();
    height = width = 0;

    const int token_number = (image_resolution / patch_size) *
                             (image_resolution / patch_size);

    struct Plane {
        int w = 0, h = 0;
        std::vector<float> px;  // (h, w, c) rows in [0,1]
    };
    std::vector<Plane> planes;

    for (const auto& path : paths) {
        int w = 0, h = 0, c = 0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
        if (!data) {
            log_error("stbi_load failed: %s", path.c_str());
            return false;
        }
        ImageBuf im;
        im.w = w;
        im.h = h;
        im.px.assign(data, data + (size_t)w * h * 3);
        stbi_image_free(data);
        crop_to_supported_aspect(im);

        const double ar = (double)im.h / (double)std::max(im.w, 1);
        // official _balanced_target_shape: w_patches = round(sqrt(T/ar)),
        // then h_patches = round(T / w_patches_FLOAT) — the float w (NOT the
        // rounded one) feeds the h division, i.e. h = round(sqrt(T*ar)).
        // Using the rounded w here shifts h by one patch for some aspect
        // ratios (e.g. ar=1.24: official 576 vs 560) and desyncs the two
        // runtimes' preprocessing for mixed-size image sets.
        const double w_patches_f = std::sqrt((double)token_number / ar);
        int target_w = 0, target_h = 0;
        if (mode == "balanced") {
            const int w_patches = std::max(1, round_half_even(w_patches_f));
            const int h_patches = std::max(
                1, round_half_even((double)token_number / w_patches_f));
            target_w = w_patches * patch_size;
            target_h = h_patches * patch_size;
        } else {
            // official _max_size_target_shape + _round_to_patch_multiple:
            // longest side = image_resolution, other side rounded to a
            // patch multiple (max(patch, round(v/patch)*patch))
            auto round_to_patch = [&](double v) {
                return std::max(patch_size,
                                round_half_even(v / patch_size) * patch_size);
            };
            if (ar >= 1.0) {
                target_h = image_resolution;
                target_w = round_to_patch(image_resolution / ar);
            } else {
                target_w = image_resolution;
                target_h = round_to_patch(image_resolution * ar);
            }
        }

        Plane pl;
        pl.w = target_w;
        pl.h = target_h;
        // stbir writes STBIR_TYPE_UINT8 bytes; stage through a byte buffer —
        // writing it straight into the float vector would reinterpret every
        // 4 bytes as one float (garbage/NaN downstream)
        std::vector<unsigned char> out8((size_t)target_w * target_h * 3);
        // Catmull-Rom filter == Keys cubic with a=-0.5, i.e. PIL "BICUBIC"
        stbir_resize(im.px.data(), im.w, im.h, 0, out8.data(), target_w,
                     target_h, 0, (stbir_pixel_layout)3, STBIR_TYPE_UINT8,
                     STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
        pl.px.resize((size_t)target_w * target_h * 3);
        for (size_t i = 0; i < pl.px.size(); i++) pl.px[i] = out8[i] / 255.0f;
        planes.push_back(std::move(pl));
    }

    // official _pad_images_to_common_size: white padding (value 1.0),
    // centered (pad_top = (maxh-h)//2 etc.). Output layout must match the
    // graph input ggml ne{W,H,3,S} == (S,3,H,W) C-order: channel before
    // height/width per frame (the previous (h,w,c) layout produced finite
    // but WRONG pixels — the model never saw the real image).
    int max_h = 0, max_w = 0;
    for (const auto& pl : planes) {
        max_h = std::max(max_h, pl.h);
        max_w = std::max(max_w, pl.w);
    }
    for (const auto& pl : planes) {
        const int pad_top = (max_h - pl.h) / 2;
        const int pad_bottom = (max_h - pl.h) - pad_top;
        const int pad_left = (max_w - pl.w) / 2;
        const int pad_right = (max_w - pl.w) - pad_left;
        const auto push_white = [&](int pixels) {
            for (int i = 0; i < pixels; i++) out.push_back(1.0f);
        };
        for (int c = 0; c < 3; c++) {
            push_white(pad_top * max_w);
            for (int y = 0; y < pl.h; y++) {
                push_white(pad_left);
                const float* row = &pl.px[(size_t)y * pl.w * 3];
                for (int x = 0; x < pl.w; x++) {
                    out.push_back(row[(size_t)x * 3 + c]);
                }
                push_white(pad_right);
            }
            push_white(pad_bottom * max_w);
        }
    }
    height = max_h;
    width = max_w;
    // (h, w, c) per image matches (H, W, C) = ggml {W, H, 3, S} rows.
    return true;
}

}  // namespace mapggml

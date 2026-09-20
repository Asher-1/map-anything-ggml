// Image loading for mapggml (convenience path).
//
// PARITY NOTE: the bit-exact input path is the raw f32 --bin feed produced by
// the official Python loader (PIL bicubic). stb's Catmull-Rom resampler is
// close to but not identical to PIL's BICUBIC; use --bin for parity runs and
// --images only for smoke tests.
#pragma once

#include <string>
#include <vector>

namespace mapggml {

// Official load_and_preprocess_images equivalent (mode = "balanced" or
// "max_size"):
//   balanced: extreme aspect ratios center-cropped into [0.5, 2.0],
//     patch-multiple target shape around (image_resolution/patch_size)^2
//     tokens; mixed sizes white-padded to a common size.
//   max_size: longest side = image_resolution, other side rounded to a
//     patch multiple; mixed sizes white-padded to a common size.
// Bicubic resize; returns (S, 3, H, W) floats in [0, 1] plus the common
// H, W.
bool load_images_official(const std::vector<std::string>& paths,
                          int image_resolution, int patch_size,
                          std::vector<float>& out, int& height, int& width,
                          const std::string& mode = "balanced");

}  // namespace mapggml

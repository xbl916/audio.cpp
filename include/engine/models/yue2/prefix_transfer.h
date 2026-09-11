#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace engine::models::yue2 {

inline bool yue2_prefix_needs_transfer(const ggml_tensor * tensor, ggml_backend_t destination) {
    if (!tensor || !tensor->buffer) {
        throw std::runtime_error("Yue2 prefix cache has no backing buffer");
    }
    const auto source_type = ggml_backend_buffer_get_type(tensor->buffer);
    // The generic CPU buffer may report a different (or no) device from the
    // selected CPU backend registry. Its default buffer is still local memory.
    if (source_type == ggml_backend_get_default_buffer_type(destination)) {
        return false;
    }
    return ggml_backend_buft_get_device(source_type) !=
           ggml_backend_get_device(destination);
}

// Synchronous host staging also works for cards without peer access (e.g. P4 +
// Turing). Bound host scratch space independently of the context length.
// The producer must finish computing the prefix before calling this function.
inline void yue2_copy_prefix_tensor(const ggml_tensor * source, ggml_tensor * destination) {
    if (!source || !destination || !source->buffer || !destination->buffer ||
        source->type != destination->type ||
        !ggml_is_contiguous(source) || !ggml_is_contiguous(destination)) {
        throw std::runtime_error("Yue2 prefix transfer requires allocated contiguous tensors of the same type");
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->ne[i] != destination->ne[i]) {
            throw std::runtime_error("Yue2 prefix transfer shape mismatch");
        }
    }
    const size_t bytes = ggml_nbytes(source);
    std::vector<uint8_t> scratch(std::min(bytes, size_t{4 * 1024 * 1024}));
    for (size_t offset = 0; offset < bytes; offset += scratch.size()) {
        const size_t count = std::min(scratch.size(), bytes - offset);
        ggml_backend_tensor_get(source, scratch.data(), offset, count);
        ggml_backend_tensor_set(destination, scratch.data(), offset, count);
    }
}

}  // namespace engine::models::yue2

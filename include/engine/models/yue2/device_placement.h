#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/options.h"

#include <map>
#include <memory>
#include <stdexcept>

namespace engine::models::yue2 {

struct Yue2DevicePlacement {
    // -1 inherits the session device. Explicit indices require CUDA.
    int ar = -1;
    int nar = -1;
    int vae = -1;
};

inline Yue2DevicePlacement parse_yue2_device_placement(
    const std::unordered_map<std::string, std::string> & options) {
    auto device = [&](const char * key) {
        const int value = runtime::parse_int_option(options, {key}).value_or(-1);
        if (value < -1) {
            throw std::runtime_error(std::string(key) + " must be -1 or a nonnegative CUDA device index");
        }
        return value;
    };
    return {device("yue2.ar_device"), device("yue2.nar_device"), device("yue2.vae_device")};
}

// Declared before the component runtimes so that backends outlive their tensors.
class Yue2DeviceContexts {
public:
    explicit Yue2DeviceContexts(core::ExecutionContext & primary) : primary_(primary) {}

    core::ExecutionContext & get(int device) {
        if (device < -1) {
            throw std::runtime_error("Yue2 component device must be -1 or a nonnegative CUDA device index");
        }
        if (device == -1) {
            return primary_;
        }
        if (primary_.backend_type() != core::BackendType::Cuda) {
            throw std::runtime_error("Yue2 component device overrides require the CUDA backend");
        }
        const auto primary_device = ggml_backend_get_device(primary_.backend());
        const auto registry = ggml_backend_dev_backend_reg(primary_device);
        if (static_cast<size_t>(device) >= ggml_backend_reg_dev_count(registry)) {
            throw std::runtime_error("Yue2 CUDA device index out of range: " + std::to_string(device));
        }
        if (ggml_backend_reg_dev_get(registry, static_cast<size_t>(device)) == primary_device) {
            return primary_;
        }
        auto & context = additional_[device];
        if (!context) {
            auto config = primary_.config();
            config.type = core::BackendType::Cuda;
            config.device = device;
            context = std::make_unique<core::ExecutionContext>(config);
        }
        return *context;
    }

private:
    core::ExecutionContext & primary_;
    std::map<int, std::unique_ptr<core::ExecutionContext>> additional_;
};

}  // namespace engine::models::yue2

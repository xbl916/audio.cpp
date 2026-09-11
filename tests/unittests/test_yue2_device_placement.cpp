#include "engine/models/yue2/device_placement.h"
#include "engine/models/yue2/prefix_transfer.h"

#include <iostream>
#include <limits>

using namespace engine;
using namespace engine::models::yue2;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F> static void rejects(F operation) {
    bool threw = false;
    try { operation(); } catch (const std::exception &) { threw = true; }
    require(threw, "expected invalid configuration/tensor to be rejected");
}

struct Tensor {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * value = nullptr;

    explicit Tensor(ggml_backend_t backend, int64_t length = 1025) {
        context = ggml_init({ggml_tensor_overhead(), nullptr, true});
        if (!context) throw std::runtime_error("test context allocation failed");
        // More than 4 MiB, with a partial final transfer chunk.
        value = ggml_new_tensor_4d(context, GGML_TYPE_F16, 1024, 2, length, 1);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) {
            ggml_free(context);
            throw std::runtime_error("test buffer allocation failed");
        }
    }
    ~Tensor() { ggml_backend_buffer_free(buffer); ggml_free(context); }
    Tensor(const Tensor &) = delete;
    Tensor & operator=(const Tensor &) = delete;
};

static void transfer(core::ExecutionContext & source_backend, core::ExecutionContext & destination_backend,
                     bool different_devices) {
    Tensor destination(destination_backend.backend());
    std::vector<uint8_t> expected(ggml_nbytes(destination.value));
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint8_t>(i * 37 + i / 257);
    {
        Tensor source(source_backend.backend());
        ggml_backend_tensor_set(source.value, expected.data(), 0, expected.size());
        require(!yue2_prefix_needs_transfer(source.value, source_backend.backend()), "same-device cache should be reused");
        require(yue2_prefix_needs_transfer(source.value, destination_backend.backend()) == different_devices,
                "incorrect cross-device cache detection");
        yue2_copy_prefix_tensor(source.value, destination.value);
        // Changing and destroying the producer must not change the consumer's copy.
        ggml_backend_buffer_clear(source.buffer, 0);
    }
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(destination.value, actual.data(), 0, actual.size());
    require(actual == expected, "prefix transfer corrupted data or borrowed producer memory");
    Tensor wrong_shape(destination_backend.backend(), 5);
    rejects([&] { yue2_copy_prefix_tensor(destination.value, wrong_shape.value); });
    rejects([&] { yue2_copy_prefix_tensor(nullptr, destination.value); });
    rejects([&] { yue2_prefix_needs_transfer(nullptr, destination_backend.backend()); });
}

int main(int argc, char ** argv) {
    try {
        const auto defaults = parse_yue2_device_placement({});
        require(defaults.ar == -1 && defaults.nar == -1 && defaults.vae == -1, "default placement changed");
        const auto split = parse_yue2_device_placement({{"yue2.ar_device", "0"}, {"yue2.nar_device", "1"}});
        require(split.ar == 0 && split.nar == 1 && split.vae == -1, "independent placement parsing failed");
        require(parse_yue2_device_placement({{"yue2.ar_device", ""}}).ar == -1,
                "empty optional value should inherit the session device");
        for (const auto * invalid : {"-2", "1.5", "1x", "2147483648"}) {
            rejects([&] { parse_yue2_device_placement({{"yue2.nar_device", invalid}}); });
        }
        core::ExecutionContext cpu({core::BackendType::Cpu, 0, 1});
        Yue2DeviceContexts cpu_contexts(cpu);
        require(&cpu_contexts.get(-1) == &cpu, "default backend was not reused");
        rejects([&] { cpu_contexts.get(0); });
        rejects([&] { cpu_contexts.get(-2); });
        transfer(cpu, cpu, false);

        // Explicit opt-in: normal CTest never allocates GPU memory.
        if (argc == 2 && std::string(argv[1]) == "--cuda") {
            core::ExecutionContext cuda({core::BackendType::Cuda, 0, 1});
            Yue2DeviceContexts contexts(cuda);
            require(&contexts.get(0) == &cuda, "primary CUDA context was not reused");
            auto & second = contexts.get(1);
            require(&contexts.get(1) == &second, "secondary context was duplicated");
            rejects([&] { contexts.get(std::numeric_limits<int>::max()); });
            transfer(cuda, second, true);
            transfer(second, cuda, true);
            transfer(cuda, cuda, false);
        }
        std::cout << "Yue2 placement and prefix transfer tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

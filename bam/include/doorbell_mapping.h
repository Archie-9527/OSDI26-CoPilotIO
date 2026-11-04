#ifndef SWARMIO_DOORBELL_MAPPING_H
#define SWARMIO_DOORBELL_MAPPING_H

#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <vector>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "nvm_types.h"

// Optional compatibility transport for emulated doorbells. Both engines must
// use the same explicitly selected mode when comparing performance.
class DoorbellMapping {
    void* registered_bar = nullptr;
    uint32_t* host = nullptr;
    uint32_t* device = nullptr;
    volatile uint32_t* real = nullptr;
    size_t stride = 0;
    std::vector<uint32_t> published;
    std::atomic<bool> stopping{false};
    std::thread worker;

public:
    DoorbellMapping() = default;
    DoorbellMapping(const DoorbellMapping&) = delete;
    DoorbellMapping& operator=(const DoorbellMapping&) = delete;

    void init(const nvm_ctrl_t* ctrl, size_t queues, bool relay) {
        if (registered_bar || host) throw std::runtime_error("doorbells already initialized");
        if (!ctrl || !ctrl->mm_ptr || !queues || queues > 64 || ctrl->dstrd > 7)
            throw std::runtime_error("invalid doorbell geometry");
        stride = size_t(1) << ctrl->dstrd; // units of uint32_t
        size_t count = (queues + 1) * 2;
        if (ctrl->mm_size < 0x1000 + ((count - 1) * stride + 1) * sizeof(uint32_t))
            throw std::runtime_error("controller mapping does not cover all doorbells");
        if (!relay) {
            unsigned flags = cudaHostRegisterIoMemory | cudaHostRegisterMapped;
            cudaError_t err = cudaHostRegister((void*)ctrl->mm_ptr, ctrl->mm_size, flags);
            if (err != cudaSuccess) {
                std::ostringstream msg;
                msg << "cudaHostRegister BAR failed: " << cudaGetErrorString(err)
                    << " (address=" << (void*)ctrl->mm_ptr << ", bytes=" << ctrl->mm_size
                    << ", flags=" << flags << "). For this virtual SSD, explicitly select --doorbell-mode relay to avoid GPU BAR registration.";
                throw std::runtime_error(msg.str());
            }
            registered_bar = (void*)ctrl->mm_ptr;
            return;
        }
        // Allocate all ordinary host state before owning a CUDA allocation.
        published.assign(count, 0);
        cudaError_t err = cudaHostAlloc((void**)&host, count * sizeof(uint32_t), cudaHostAllocMapped);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        err = cudaHostGetDevicePointer((void**)&device, host, 0);
        if (err != cudaSuccess) {
            cudaFreeHost(host); host = nullptr; device = nullptr;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        memset(host, 0, count * sizeof(uint32_t));
        real = (volatile uint32_t*)((char*)ctrl->mm_ptr + 0x1000);
    }

    volatile uint32_t* device_doorbells() const { return device; }

    // No admin doorbell forwarding: queue 0 remains managed by the CPU API.
    // Called by one CPU thread. GPU producers fence their SQ/CQ writes before
    // publishing these mapped host words with system scope stores.
    void poll_once() {
        for (size_t i = 2; i < published.size(); ++i) {
            uint32_t value = static_cast<volatile uint32_t*>(host)[i];
            if (value != published[i]) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
                real[i * stride] = value;
                published[i] = value;
            }
        }
    }

    void start() {
        if (!host || worker.joinable()) return;
        stopping.store(false, std::memory_order_release);
        worker = std::thread([this] {
            while (!stopping.load(std::memory_order_acquire)) {
                poll_once();
#if defined(__x86_64__) || defined(__i386__)
                __asm__ volatile("pause" ::: "memory");
#else
                std::this_thread::yield();
#endif
            }
        });
    }

    void destroy() {
        stopping.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
        // Caller has synchronized the GPU; forward the final CQ head as well.
        if (host) { poll_once(); cudaFreeHost(host); }
        if (registered_bar) cudaHostUnregister(registered_bar);
        host = nullptr; device = nullptr; registered_bar = nullptr; real = nullptr;
        published.clear();
    }
    ~DoorbellMapping() { destroy(); }
};
#endif

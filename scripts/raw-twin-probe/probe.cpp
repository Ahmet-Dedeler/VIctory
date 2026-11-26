#define INITGUID
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;
using Resource = ComPtr<ID3D12Resource>;
constexpr auto heap_type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
constexpr unsigned words = 16384;
constexpr unsigned cases = 64;
constexpr unsigned requests = 27500;

void check(HRESULT result, const char *operation) {
    if (FAILED(result)) {
        std::fprintf(stderr, "%s failed: 0x%08lx\n", operation, static_cast<unsigned long>(result));
        throw std::runtime_error(operation);
    }
}

struct Heap {
    ComPtr<ID3D12DescriptorHeap> object;
    UINT step;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_base;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_base;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(unsigned slot) const {
        return {cpu_base.ptr + SIZE_T(step) * slot};
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(unsigned slot) const {
        return {gpu_base.ptr + UINT64(step) * slot};
    }
};

Heap heap(ID3D12Device *device, unsigned count, bool visible) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = heap_type;
    desc.NumDescriptors = count;
    desc.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    Heap result{};
    result.step = device->GetDescriptorHandleIncrementSize(heap_type);
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(result.object.GetAddressOf())), "heap");
    result.cpu_base = result.object->GetCPUDescriptorHandleForHeapStart();
    if (visible) result.gpu_base = result.object->GetGPUDescriptorHandleForHeapStart();
    return result;
}

Resource buffer(ID3D12Device *device, D3D12_HEAP_TYPE type, unsigned bytes, D3D12_RESOURCE_STATES state,
                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    Resource result;
    check(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                          IID_PPV_ARGS(result.GetAddressOf())), "buffer");
    return result;
}

void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

void srv(ID3D12Device *device, ID3D12Resource *buffer, unsigned first, bool raw,
         D3D12_CPU_DESCRIPTOR_HANDLE dst) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R32_UINT;
    desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Buffer.FirstElement = first;
    desc.Buffer.NumElements = 4;
    desc.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
    device->CreateShaderResourceView(buffer, &desc, dst);
}

void uav(ID3D12Device *device, ID3D12Resource *buffer, unsigned first, bool raw,
         D3D12_CPU_DESCRIPTOR_HANDLE dst, unsigned count = 4) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R32_UINT;
    desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    desc.Buffer.FirstElement = first;
    desc.Buffer.NumElements = count;
    desc.Buffer.Flags = raw ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView(buffer, nullptr, &desc, dst);
}

void twins(ID3D12Device *device, const Heap &stage, ID3D12Resource *source,
           ID3D12Resource *writable, unsigned first, bool raw, bool reuse) {
    srv(device, source, first, raw, stage.cpu(0));
    uav(device, writable, first, raw, stage.cpu(2));
    if (raw && reuse) {
        device->CopyDescriptorsSimple(1, stage.cpu(1), stage.cpu(0), heap_type);
        device->CopyDescriptorsSimple(1, stage.cpu(3), stage.cpu(2), heap_type);
    } else {
        srv(device, source, first, true, stage.cpu(1));
        uav(device, writable, first, true, stage.cpu(3));
    }
}

ComPtr<ID3D12RootSignature> signature(ID3D12Device *device) {
    D3D12_DESCRIPTOR_RANGE ranges[3]{};
    D3D12_ROOT_PARAMETER params[4]{};
    for (unsigned i = 0; i < 3; ++i) {
        ranges[i].RangeType = i == 0 ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[i].NumDescriptors = 1;
        ranges[i].BaseShaderRegister = i == 2 ? 1 : 0;
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable = {1, &ranges[i]};
    }
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants.Num32BitValues = 1;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 4;
    desc.pParameters = params;
    ComPtr<ID3DBlob> bytes, error;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, bytes.GetAddressOf(),
                                      error.GetAddressOf()), "serialize signature");
    ComPtr<ID3D12RootSignature> result;
    check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(),
                                       IID_PPV_ARGS(result.GetAddressOf())), "root signature");
    return result;
}

ComPtr<ID3D12PipelineState> pipeline(ID3D12Device *device, ID3D12RootSignature *root) {
    std::ifstream file("verify.dxil", std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("open verify.dxil");
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size())) throw std::runtime_error("read verify.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {bytes.data(), bytes.size()};
    ComPtr<ID3D12PipelineState> result;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(result.GetAddressOf())), "pipeline");
    return result;
}

Resource pattern(ID3D12Device *device, uint32_t prefix) {
    auto resource = buffer(device, D3D12_HEAP_TYPE_UPLOAD, words * 4, D3D12_RESOURCE_STATE_GENERIC_READ);
    void *mapped = nullptr;
    D3D12_RANGE no_read{0, 0};
    check(resource->Map(0, &no_read, &mapped), "map upload");
    auto *data = static_cast<uint32_t *>(mapped);
    for (unsigned i = 0; i < words; ++i) data[i] = prefix | i;
    resource->Unmap(0, nullptr);
    return resource;
}

void verify(ID3D12Device *device, const Heap &stage,
            const std::array<Resource, 2> &sources, const std::array<Resource, 2> &writes) {
    auto visible = heap(device, 4 * cases + 1, true);
    auto output = buffer(device, D3D12_HEAP_TYPE_DEFAULT, cases * 16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, cases * 16, D3D12_RESOURCE_STATE_COPY_DEST);
    auto mutated = buffer(device, D3D12_HEAP_TYPE_READBACK, 2 * words * 4, D3D12_RESOURCE_STATE_COPY_DEST);
    for (unsigned i = 0; i < cases; ++i) {
        // Reuse the same staging handles with another backing resource,
        // range, and view type for the next case before the GPU reads any.
        const unsigned bank = (i / 4) % 2;
        const bool raw = i % 4 < 2;
        const bool reuse = i % 2 != 0;
        const unsigned first = 4 * i;
        auto *source = sources[bank].Get();
        auto *writable = writes[bank].Get();
        if (i % 16 < 8) {
            twins(device, stage, source, writable, first, raw, reuse);
            device->CopyDescriptorsSimple(1, visible.cpu(first + 1), stage.cpu(1), heap_type);
            device->CopyDescriptorsSimple(1, visible.cpu(first + 3), stage.cpu(3), heap_type);
        } else {
            // Direct-making fallback: shader-visible slots cannot source copies.
            srv(device, source, first, raw, visible.cpu(first));
            srv(device, source, first, true, visible.cpu(first + 1));
            uav(device, writable, first, raw, visible.cpu(first + 2));
            uav(device, writable, first, true, visible.cpu(first + 3));
        }
    }
    uav(device, output.Get(), 0, true, visible.cpu(4 * cases), cases * 4);

    auto root = signature(device);
    auto state = pipeline(device, root.Get());
    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "allocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), state.Get(),
                                    IID_PPV_ARGS(list.GetAddressOf())), "list");
    std::array<Resource, 2> initial = {pattern(device, 0x31000000), pattern(device, 0x42000000)};
    for (unsigned bank = 0; bank < writes.size(); ++bank) {
        list->CopyBufferRegion(writes[bank].Get(), 0, initial[bank].Get(), 0, words * 4);
        transition(list.Get(), writes[bank].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12DescriptorHeap *heaps[] = {visible.object.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootDescriptorTable(2, visible.gpu(4 * cases));
    for (unsigned i = 0; i < cases; ++i) {
        list->SetComputeRootDescriptorTable(0, visible.gpu(4 * i + 1));
        list->SetComputeRootDescriptorTable(1, visible.gpu(4 * i + 3));
        list->SetComputeRoot32BitConstant(3, i, 0);
        list->Dispatch(1, 1, 1);
    }
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(readback.Get(), output.Get());
    for (unsigned bank = 0; bank < writes.size(); ++bank) {
        transition(list.Get(), writes[bank].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(mutated.Get(), bank * words * 4, writes[bank].Get(), 0, words * 4);
    }
    check(list->Close(), "close");
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())), "queue");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())), "fence");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("event");
    ID3D12CommandList *lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    check(queue->Signal(fence.Get(), 1), "signal");
    check(fence->SetEventOnCompletion(1, event), "fence event");
    DWORD waited = WaitForSingleObject(event, 15000);
    CloseHandle(event);
    if (waited != WAIT_OBJECT_0) throw std::runtime_error("GPU wait");
    check(device->GetDeviceRemovedReason(), "device");

    void *mapped = nullptr;
    check(readback->Map(0, nullptr, &mapped), "readback map");
    auto *values = static_cast<const uint32_t *>(mapped);
    for (unsigned i = 0; i < cases; ++i) {
        const unsigned bank = (i / 4) % 2;
        const auto source_prefix = bank == 0 ? 0x11000000u : 0x22000000u;
        const auto write_prefix = bank == 0 ? 0x31000000u : 0x42000000u;
        const std::array<uint32_t, 4> expected = {
            source_prefix | (4 * i), source_prefix | (4 * i + 1),
            write_prefix | (4 * i), source_prefix ^ write_prefix,
        };
        if (!std::equal(expected.begin(), expected.end(), values + 4 * i)) {
            std::fprintf(stderr, "case %u mismatch: %08x %08x %08x %08x\n",
                         i, values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3]);
            throw std::runtime_error("descriptor readback mismatch");
        }
    }
    readback->Unmap(0, nullptr);
    check(mutated->Map(0, nullptr, &mapped), "mutated map");
    values = static_cast<const uint32_t *>(mapped);
    for (unsigned bank = 0; bank < writes.size(); ++bank) {
        const auto source_prefix = bank == 0 ? 0x11000000u : 0x22000000u;
        const auto write_prefix = bank == 0 ? 0x31000000u : 0x42000000u;
        for (unsigned i = 0; i < words; ++i) {
            const bool changed = i < 4 * cases && i % 4 == 0 && (i / 16) % 2 == bank;
            const uint32_t expected = changed ? source_prefix ^ write_prefix : write_prefix | i;
            if (values[bank * words + i] != expected) throw std::runtime_error("UAV wrote the wrong range");
        }
    }
    mutated->Unmap(0, nullptr);
    std::puts("PASS 64 SRV/UAV pairs, exact raw copies, typed and visible-heap fallbacks, stage reused before dispatch");
}

void measure(ID3D12Device *device, const Heap &stage, ID3D12Resource *source, ID3D12Resource *writable) {
    using clock = std::chrono::steady_clock;
    for (unsigned hit_rate : {100u, 50u, 0u}) {
        std::array<std::vector<double>, 2> times;
        for (unsigned sample = 0; sample < 31; ++sample) {
            for (unsigned order = 0; order < 2; ++order) {
                const bool reuse = (sample + order) % 2;
                const auto start = clock::now();
                for (unsigned i = 0; i < requests; ++i) {
                    const bool raw = (i % 100) < hit_rate;
                    twins(device, stage, source, writable, (i % (words / 4)) * 4, raw, reuse);
                }
                times[reuse].push_back(std::chrono::duration<double, std::milli>(clock::now() - start).count());
            }
        }
        for (unsigned arm = 0; arm < 2; ++arm) {
            auto &v = times[arm];
            std::sort(v.begin(), v.end());
            std::printf("eligible %u%%, %s, %u descriptor pairs: %.3f ms p50 / %.3f ms p95\n",
                        hit_rate, arm ? "reuse" : "create", requests, v[15], v[29]);
        }
    }
}

int main() {
    try {
        ComPtr<ID3D12Device> device;
        check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())), "device");
        const auto stage = heap(device.Get(), 4, false);
        const std::array<Resource, 2> sources = {pattern(device.Get(), 0x11000000), pattern(device.Get(), 0x22000000)};
        std::array<Resource, 2> writes;
        for (auto &resource : writes) {
            resource = buffer(device.Get(), D3D12_HEAP_TYPE_DEFAULT, words * 4, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        verify(device.Get(), stage, sources, writes);
        measure(device.Get(), stage, sources[0].Get(), writes[0].Get());
        check(device->GetDeviceRemovedReason(), "device after measurement");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

// SPDX-License-Identifier: Apache-2.0
// Derived from devin-lai/DeepSeek-V4.1-Flash-Accel at 4a4e88e.
// Modified for HIP/gfx1201: generic host VMM, device guards and per-run cleanup.
//
// Partially resident tensors: one contiguous device address range whose pages
// are backed by a mixture of GPU memory and pinned host memory.
//
// The offloaded expert weights of DeepSeek-V4.1-Flash are read as whole packed
// matrices by one packed GEMM per projection, so the kernel needs a single
// base pointer with the stock [E, N, K] layout. The HIP virtual memory
// management API supplies exactly that: reserve a virtual range, map some of
// its pages to `hipMemLocationTypeDevice` memory and the rest to
// `hipMemLocationTypeHost` memory, and grant the GPU read/write access
// to the whole range. Hot experts are then read at GPU memory bandwidth and
// cold experts over PCIe, with no kernel change and no change to the values.
//
// Nothing here converts, re-packs or re-quantizes anything; it moves bytes.

#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

void check(hipError_t status, const char *what) {
  TORCH_CHECK(status == hipSuccess, what, " failed: ", hipGetErrorString(status));
}

hipMemAllocationProp device_prop(int device) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  return prop;
}

hipMemAllocationProp host_prop(int numa_node, int location_type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = static_cast<hipMemLocationType>(location_type);
  // The generic host location has no node to name; the NUMA one does.
  prop.location.id = (location_type == hipMemLocationTypeHost) ? 0 : numa_node;
  return prop;
}

size_t granularity_of(const hipMemAllocationProp &prop) {
  size_t value = 0;
  check(hipMemGetAllocationGranularity(&value, &prop,
                                      hipMemAllocationGranularityRecommended),
        "hipMemGetAllocationGranularity");
  return value;
}

// Owns one virtual range and the physical handles mapped into it. The tensor
// handed back to Python keeps a shared_ptr to this, so the mapping outlives
// every view of the weights and is released when the last one goes away.
struct Mapping {
  void* base = nullptr;
  size_t reserved = 0;
  int device = 0;
  std::vector<hipMemGenericAllocationHandle_t> handles;
  std::vector<std::pair<void*, size_t>> runs;
  ~Mapping() {
    int previous = 0;
    hipGetDevice(&previous);
    hipSetDevice(device);
    // VMM storage is outside PyTorch's caching allocator. Wait before unmapping
    // immutable weights potentially referenced by queued work on this device.
    hipDeviceSynchronize();
    for (auto run : runs) hipMemUnmap(run.first, run.second);
    if (base) hipMemAddressFree(base, reserved);
    for (auto handle : handles) hipMemRelease(handle);
    hipSetDevice(previous);
  }
};

}  // namespace

int64_t host_location_type(int64_t device, int64_t numa_node);

int64_t page_size(int64_t device, int64_t numa_node) {
  const int location = static_cast<int>(host_location_type(device, numa_node));
  TORCH_CHECK(location != 0, "no host memory location works through the HIP "
                             "virtual memory API on device ", device);
  size_t device_grain = granularity_of(device_prop(static_cast<int>(device)));
  size_t host_grain =
      granularity_of(host_prop(static_cast<int>(numa_node), location));
  return static_cast<int64_t>(std::max(device_grain, host_grain));
}

// Attribute queries disagree with drivers often enough that the only reliable
// answer is an allocation. Probe generic host VMM; this HIP port does not
// claim NUMA-specific placement.
static int probe_host_location(int device, int numa_node) {
  const int candidates[] = {hipMemLocationTypeHost};
  for (int type : candidates) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = static_cast<hipMemLocationType>(type);
    prop.location.id = (type == hipMemLocationTypeHost) ? 0 : numa_node;
    size_t grain = 0;
    if (hipMemGetAllocationGranularity(&grain, &prop,
                                      hipMemAllocationGranularityRecommended) !=
            hipSuccess ||
        grain == 0) {
      continue;
    }
    hipMemGenericAllocationHandle_t handle = 0;
    if (hipMemCreate(&handle, grain, &prop, 0) == hipSuccess) {
      hipMemRelease(handle);
      return type;
    }
  }
  return 0;
}

int64_t host_location_type(int64_t device, int64_t numa_node) {
  c10::DeviceGuard guard(at::Device(at::kCUDA, device));
  return probe_host_location(static_cast<int>(device), 0);
}

bool host_vmm_supported(int64_t device) {
  return host_location_type(device, 0) != 0;
}

// `source` is the resident or UVA tensor as loaded; `page_on_device` has one
// entry per page of it, true where that page should stay in GPU memory. The
// returned tensor has the same sizes, strides, dtype and device, and holds the
// same bytes.
torch::Tensor make_partially_resident(torch::Tensor source,
                                      torch::Tensor page_on_device,
                                      int64_t numa_node) {
  TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");
  TORCH_CHECK(source.numel() > 0, "source must not be empty");
  TORCH_CHECK(page_on_device.device().is_cpu() &&
                  page_on_device.scalar_type() == at::kBool && page_on_device.is_contiguous(),
              "page_on_device must be a CPU bool tensor");

  const int device = static_cast<int>(source.device().index());
  c10::DeviceGuard guard(source.device());
  const int location = static_cast<int>(host_location_type(device, numa_node));
  TORCH_CHECK(location != 0, "no host memory location works through the HIP "
                             "virtual memory API on device ", device);
  const auto prop_device = device_prop(device);
  const auto prop_host = host_prop(static_cast<int>(numa_node), location);
  const size_t grain =
      std::max(granularity_of(prop_device), granularity_of(prop_host));

  const size_t bytes = source.nbytes();
  const size_t pages = (bytes + grain - 1) / grain;
  TORCH_CHECK(static_cast<size_t>(page_on_device.numel()) == pages,
              "page_on_device has ", page_on_device.numel(), " entries, need ",
              pages);

  const bool *flags = page_on_device.data_ptr<bool>();

  auto mapping = std::make_shared<Mapping>();
  mapping->device = device;
  mapping->reserved = pages * grain;
  check(hipMemAddressReserve(&mapping->base, mapping->reserved, grain, 0, 0),
        "hipMemAddressReserve");

  // One physical allocation per maximal same-kind run: `hipMemMap` only accepts
  // offset zero into a handle, so a single big allocation cannot be carved up.
  size_t index = 0;
  while (index < pages) {
    const bool kind = flags[index];
    size_t run = 1;
    while (index + run < pages && flags[index + run] == kind) {
      ++run;
    }
    const size_t length = run * grain;
    hipMemGenericAllocationHandle_t handle = 0;
    check(hipMemCreate(&handle, length, kind ? &prop_device : &prop_host, 0),
          kind ? "hipMemCreate(device)" : "hipMemCreate(host)");
    mapping->handles.push_back(handle);
    check(hipMemMap(static_cast<char*>(mapping->base) + index * grain, length, 0, handle, 0),
          kind ? "hipMemMap(device)" : "hipMemMap(host)");
    mapping->runs.emplace_back(static_cast<char*>(mapping->base) + index * grain, length);
    index += run;
  }

  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  check(hipMemSetAccess(mapping->base, mapping->reserved, &access, 1),
        "hipMemSetAccess");

  check(hipMemcpy(reinterpret_cast<void *>(mapping->base),
                            source.data_ptr(), bytes, hipMemcpyDefault), "copy packed bytes");

  auto sizes = source.sizes().vec();
  auto strides = source.strides().vec();
  auto options = source.options();
  // `from_blob` normally infers the device from the pointer, and a virtual
  // range whose first page is host backed does not report the GPU that owns
  // the rest of it. Name the device instead of letting it be inferred.
  return at::from_blob(
      reinterpret_cast<void *>(mapping->base), sizes, strides,
      [mapping](void *) mutable { mapping.reset(); }, options,
      at::Device(at::kCUDA, device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("page_size", &page_size, "VMM allocation granularity in bytes");
  m.def("host_vmm_supported", &host_vmm_supported,
        "whether this device can map host memory through VMM");
  m.def("host_location_type", &host_location_type,
        "hipMemLocationType the driver accepts for host allocations, 0 if none");
  m.def("make_partially_resident", &make_partially_resident,
        "copy a tensor into a mixed device/host virtual range");
}

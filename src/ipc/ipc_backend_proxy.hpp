/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_IPC_BACKEND_PROXY_HPP_
#define LIBRARY_SRC_IPC_BACKEND_PROXY_HPP_

#include "../device_proxy.hpp"
#include "../atomic_return.hpp"

namespace rocshmem {

struct IPCBackendRegister {
  char *g_ret{nullptr};
  atomic_ret_t *atomic_ret{nullptr};
  SymmetricHeap *heap_ptr{nullptr};
};

template <typename ALLOCATOR>
class IPCBackendProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, IPCBackendRegister>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  IPCBackendProxy() { new (proxy_.get()) IPCBackendRegister(); }

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~IPCBackendProxy() { proxy_.get()->~IPCBackendRegister(); }

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ IPCBackendRegister *get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};
};

using IPCBackendProxyT = IPCBackendProxy<HIPHostAllocator>;

}  // namespace rocshmem

#endif  // #define LIBRARY_SRC_IPC_BACKEND_PROXY_HPP_
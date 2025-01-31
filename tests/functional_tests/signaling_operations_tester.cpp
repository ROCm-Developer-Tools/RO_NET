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

#include "signaling_operations_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void SignalingOperationsTest(int loop, int skip,
                              uint64_t *start_time, uint64_t *end_time,
                              int wf_size, char *s_buf, char *r_buf, int size,
                              uint64_t *sig_addr, uint64_t *fetched_value,
                              TestType type, ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t signal = 0;
  int sig_op = ROCSHMEM_SIGNAL_SET;
  int idx = 0;
  int wf_id = 0;
  int wg_id = get_flat_grid_id();
  int wg_offset = 0;

  switch (type) {
    case PutSignalTestType:
    case PutSignalNBITestType:
    case SignalFetchTestType:
      /**
       * Calculate start index for each thread within the grid
       */
      idx = get_flat_id();
      break;
    case WGPutSignalTestType:
    case WGPutSignalNBITestType:
    case WGSignalFetchTestType:
      /**
       * Calculate start index for each work group
       */
      idx = get_flat_grid_id();
      break;
    case WAVEPutSignalTestType:
    case WAVEPutSignalNBITestType:
    case WAVESignalFetchTestType:
      /**
       * Calculate start index for each wavefront
       */
      wf_id = get_flat_block_id() / wf_size;
      wg_offset = get_flat_grid_id() *
                  ((get_flat_block_size() - 1) / wf_size + 1);
      idx = wf_id + wg_offset;
      break;
    default:
      break;
  }

  s_buf += size * idx;
  r_buf += size * idx;
  sig_addr += idx;
  fetched_value += idx;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case PutSignalTestType:
        rocshmem_ctx_putmem_signal(ctx, r_buf, s_buf, size, sig_addr,
                                   signal, sig_op, 1);
        break;
      case WGPutSignalTestType:
        rocshmem_ctx_putmem_signal_wg(ctx, r_buf, s_buf, size, sig_addr,
                                      signal, sig_op, 1);
        break;
      case WAVEPutSignalTestType:
        rocshmem_ctx_putmem_signal_wave(ctx, r_buf, s_buf, size, sig_addr,
                                        signal, sig_op, 1);
        break;
      case PutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi(ctx, r_buf, s_buf, size, sig_addr,
                                       signal, sig_op, 1);
        break;
      case WGPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wg(ctx, r_buf, s_buf, size, sig_addr,
                                          signal, sig_op, 1);
        break;
      case WAVEPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wave(ctx, r_buf, s_buf, size, sig_addr,
                                            signal, sig_op, 1);
        break;
      case SignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch(sig_addr);
        break;
      case WGSignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wg(sig_addr);
        break;
      case WAVESignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wave(sig_addr);
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SignalingOperationsTester::SignalingOperationsTester(TesterArguments args)
  : Tester(args) {

  type = (TestType)args.algorithm;

  switch (type) {
    case PutSignalTestType:
    case PutSignalNBITestType:
      buff_size = args.max_msg_size * args.num_wgs * args.wg_size;
      num_signals = args.num_wgs * args.wg_size;
      break;
    case WAVEPutSignalTestType:
    case WAVEPutSignalNBITestType:
      buff_size = args.max_msg_size * args.num_wgs * num_warps;
      num_signals = args.num_wgs * num_warps;
      break;
    case WGPutSignalTestType:
    case WGPutSignalNBITestType:
      buff_size = args.max_msg_size * args.num_wgs;
      num_signals = args.num_wgs;
      break;
    case SignalFetchTestType:
      num_signals = args.num_wgs * args.wg_size;
      break;
    case WAVESignalFetchTestType:
      num_signals = args.num_wgs * num_warps;
      break;
    case WGSignalFetchTestType:
      num_signals = args.num_wgs;
      break;
    default:
      break;
  }

  s_buf = (char *)rocshmem_malloc(buff_size);
  r_buf = (char *)rocshmem_malloc(buff_size);
  sig_addr = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * num_signals);
  CHECK_HIP(hipMallocManaged(&fetched_value, sizeof(uint64_t) * num_signals,
                             hipMemAttachHost));

}

SignalingOperationsTester::~SignalingOperationsTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  rocshmem_free(sig_addr);
  CHECK_HIP(hipFree(fetched_value));
}

void SignalingOperationsTester::resetBuffers(uint64_t size) {
  memset(s_buf, '0', buff_size);
  memset(r_buf, '1', buff_size);
  for (int i = 0; i < num_signals; i++) {
    fetched_value[i] = -1;
    sig_addr[i] = args.myid + 123;
  }
}

void SignalingOperationsTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                             int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SignalingOperationsTest, gridSize, blockSize,
                     shared_bytes, stream, loop, args.skip, start_time,
                     end_time, wf_size, s_buf, r_buf, size, sig_addr,
                     fetched_value, _type, _shmem_context);

  num_msgs = (loop + args.skip);
  num_timed_msgs = loop;

  switch (type) {
    case PutSignalTestType:
    case PutSignalNBITestType:
    case SignalFetchTestType:
      num_msgs *= gridSize.x * blockSize.x;
      num_timed_msgs *= gridSize.x * blockSize.x;
      break;
    case WAVEPutSignalTestType:
    case WAVEPutSignalNBITestType:
    case WAVESignalFetchTestType:
      num_msgs *= gridSize.x * num_warps;
      num_timed_msgs *= gridSize.x * num_warps;
      break;
    case WGPutSignalTestType:
    case WGPutSignalNBITestType:
    case WGSignalFetchTestType:
      num_msgs *= gridSize.x;
      num_timed_msgs *= gridSize.x;
      break;
    default:
      num_msgs = (loop + args.skip);
      num_timed_msgs = loop;
      break;
  }
}

void SignalingOperationsTester::verifyResults(uint64_t size) {
  int check_data_id = (_type == PutSignalTestType ||
                       _type == PutSignalNBITestType ||
                       _type == WAVEPutSignalTestType ||
                       _type == WAVEPutSignalNBITestType ||
                       _type == WGPutSignalTestType ||
                       _type == WGPutSignalNBITestType)
                    ? 1 : -1; // do not check if it doesn't match a test

  int check_fetched_value_id = (_type == SignalFetchTestType ||
                                _type == WAVESignalFetchTestType ||
                                _type == WGSignalFetchTestType)
                             ? 0 : -1; // do not check if it doesn't match a test

  switch (type) {
    case PutSignalTestType:
    case PutSignalNBITestType:
      size *= args.num_wgs * args.wg_size;
      break;
    case WAVEPutSignalTestType:
    case WAVEPutSignalNBITestType:
      size *= args.num_wgs * num_warps;
      break;
    case WGPutSignalTestType:
    case WGPutSignalNBITestType:
      size *= args.num_wgs;
      break;
    default:
      break;
  }

  if (args.myid == check_data_id) {
    for (uint64_t i = 0; i < size; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %lu\n", i);
        fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
        exit(-1);
      }
    }
    for (int i = 0; i < num_signals; i++) {
      if (sig_addr[i] != 0) {
        fprintf(stderr, "Signal Value %lu, Expected 0\n", sig_addr[i]);
        exit(-1);
      }
    }
  }

  if (args.myid == check_fetched_value_id) {
    uint64_t expected_value = (args.myid + 123);
    for (int i = 0; i < num_signals; i++) {
      if (fetched_value[i] != expected_value) {
        fprintf(stderr, "Fetched Value %lu, Expected %lu\n", fetched_value[i], expected_value);
        exit(-1);
      }
    }
  }
}

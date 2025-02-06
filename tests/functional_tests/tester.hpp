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

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "tester_arguments.hpp"
#include "../src/util.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/
enum TestType {
  GetTestType = 0,
  GetNBITestType = 1,
  PutTestType = 2,
  PutNBITestType = 3,
  GetSwarmTestType = 4,
  AMO_FAddTestType = 5,
  AMO_FIncTestType = 6,
  AMO_FetchTestType = 7,
  AMO_FCswapTestType = 8,
  AMO_AddTestType = 9,
  AMO_IncTestType = 10,
  AMO_CswapTestType = 11,
  InitTestType = 12,
  PingPongTestType = 13,
  RandomAccessTestType = 14,
  BarrierAllTestType = 15,
  SyncAllTestType = 16,
  SyncTestType = 17,
  CollectTestType = 18,
  FCollectTestType = 19,
  AllToAllTestType = 20,
  AllToAllsTestType = 21,
  ShmemPtrTestType = 22,
  PTestType = 23,
  GTestType = 24,
  WGGetTestType = 25,
  WGGetNBITestType = 26,
  WGPutTestType = 27,
  WGPutNBITestType = 28,
  WAVEGetTestType = 29,
  WAVEGetNBITestType = 30,
  WAVEPutTestType = 31,
  WAVEPutNBITestType = 32,
  TeamBroadcastTestType = 33,
  TeamReductionTestType = 34,
  TeamCtxGetTestType = 35,
  TeamCtxGetNBITestType = 36,
  TeamCtxPutTestType = 37,
  TeamCtxPutNBITestType = 38,
  TeamCtxInfraTestType = 39,
  PutNBIMRTestType = 40,
  AMO_SetTestType = 41,
  AMO_SwapTestType = 42,
  AMO_FetchAndTestType = 43,
  AMO_FetchOrTestType = 44,
  AMO_FetchXorTestType = 45,
  AMO_AndTestType = 46,
  AMO_OrTestType = 47,
  AMO_XorTestType = 48,
  PingAllTestType = 49,
  PutSignalTestType = 50,
  WGPutSignalTestType = 51,
  WAVEPutSignalTestType = 52,
  PutSignalNBITestType = 53,
  WGPutSignalNBITestType = 54,
  WAVEPutSignalNBITestType = 55,
  SignalFetchTestType = 56,
  WGSignalFetchTestType = 57,
  WAVESignalFetchTestType = 58,
};

enum OpType { PutType = 0, GetType = 1 };

typedef int ShmemContextType;

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
class Tester {
 public:
  explicit Tester(TesterArguments args);
  virtual ~Tester();

  void execute();

  static std::vector<Tester *> create(TesterArguments args);

 protected:
  virtual void resetBuffers(uint64_t size) = 0;

  virtual void preLaunchKernel() {}

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) = 0;

  virtual void postLaunchKernel() {}

  virtual void verifyResults(uint64_t size) = 0;

  int num_msgs = 0;
  int num_timed_msgs = 0;
  int num_warps = 0;
  int bw_factor = 1;
  int device_id = 0;
  int wall_clk_rate = 0; //in kilohertz
  int wf_size = 0;

  TesterArguments args;

  TestType _type;
  ShmemContextType _shmem_context = 8;  // SHMEM_CTX_WP_PRIVATE

  hipStream_t stream;
  hipDeviceProp_t deviceProps;

  uint64_t *timer = nullptr;
  uint64_t *start_time = nullptr;
  uint64_t *end_time = nullptr;
  uint64_t min_start_time = 0;
  uint64_t max_end_time = 0;
  uint32_t num_timers = 0;

 private:
  bool _print_header = 1;
  void print(uint64_t size);

  void barrier();

  double gpuCyclesToMicroseconds(uint64_t cycles);

  double timerAvgInMicroseconds();

  bool peLaunchesKernel();

  hipEvent_t start_event;
  hipEvent_t stop_event;
};

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#endif /* _TESTER_HPP */

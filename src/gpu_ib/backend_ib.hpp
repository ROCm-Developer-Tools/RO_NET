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

#ifndef LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_
#define LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_

#include "../backend_bc.hpp"
#include "../containers/free_list_impl.hpp"
#include "network_policy.hpp"
#include "../hdp_policy.hpp"
#include "../hdp_proxy.hpp"
#include "../memory/hip_allocator.hpp"

namespace rocshmem {

class HostInterface;

/**
 * @class GPUIBBackend backend.hpp
 * @brief InfiniBand specific backend.
 *
 * The InfiniBand (GPUIB) backend enables the device to enqueue network
 * requests to InfiniBand queues (with minimal host intervention). The setup
 * requires some effort from the host, but the device is able to craft
 * InfiniBand requests and send them on its own.
 */
class GPUIBBackend : public Backend {
 public:
  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit GPUIBBackend(MPI_Comm comm);

  /**
   * @copydoc Backend::~Backend()
   */
  virtual ~GPUIBBackend();

  /**
   * @brief Abort the application.
   *
   * @param[in] status Exit code.
   *
   * @return void.
   *
   * @note This routine terminates the entire application.
   */
  void global_exit(int status) override;

  /**
   * @copydoc Backend::create_new_team
   */
  void create_new_team(Team *parent_team, TeamInfo *team_info_wrt_parent,
                       TeamInfo *team_info_wrt_world, int num_pes,
                       int my_pe_in_new_team, MPI_Comm team_comm,
                       rocshmem_team_t *new_team) override;

  /**
   * @copydoc Backend::team_destroy(rocshmem_team_t)
   */
  void team_destroy(rocshmem_team_t team) override;

  /**
   * @copydoc Backend::ctx_create
   */
  void ctx_create(int64_t options, void **ctx) override;

  __device__ bool create_ctx(int64_t options, rocshmem_ctx_t *ctx);

  /**
   * @copydoc Backend::ctx_destroy
   */
  void ctx_destroy(Context *ctx) override;

  /**
   * @copydoc Backend::ctx_destroy
   */
  __device__ void destroy_ctx(rocshmem_ctx_t *ctx);

 protected:
  /**
   * @copydoc Backend::dump_backend_stats()
   */
  void dump_backend_stats() override;

  /**
   * @copydoc Backend::reset_backend_stats()
   */
  void reset_backend_stats() override;

  /**
   * @brief spawn a new thread to perform the rest of initialization
   */
  std::thread thread_spawn(GPUIBBackend *b);

  /**
   * @brief overheads for helper thread to run
   *
   * @param[in] the thread needs access to the class
   *
   * @return void
   */
  void thread_func_internal(GPUIBBackend *b);

  /**
   * @brief initialize MPI.
   *
   * GPUIB relies on MPI just to exchange the connection information.
   *
   * todo: remove the dependency on MPI and make it generic to PMI-X or just
   * to OpenSHMEM to have support for both CPU and GPU
   */
  void init_mpi_once(MPI_Comm comm);

  /**
   * @brief init the network support
   */
  void initialize_network();

  /**
   * @brief Invokes the IPC policy class initialization method.
   *
   * This method delegates Inter Process Communication (IPC)
   * initialization to the appropriate policy class. The initialization
   * needs to be exposed to the Backed due to initialization ordering
   * constraints. (The symmetric heaps needs to be allocated and
   * initialized before this method can be called.)
   *
   * The policy class encapsulates what the initialization process so
   * refer to that class for more details.
   */
  void initialize_ipc();

  /**
   * @brief Allocate and initialize the ROCSHMEM_CTX_DEFAULT variable.
   *
   * @todo The default_ctx member looks unused after it is copied into
   * the ROCSHMEM_CTX_DEFAULT variable.
   */
  void setup_default_ctx();
  void setup_ctxs();

  /**
   * @brief Allocate and initialize the default context for host
   * operations.
   */
  void setup_default_host_ctx();

  /**
   * @brief Allocate and initialize team world.
   */
  void setup_team_world();

  /**
   * @brief Initialize the resources required to support teams
   */
  void teams_init();

  /**
   * @brief Destruct the resources required to support teams
   */
  void teams_destroy();

  /**
   * @brief Allocate and initialize barrier operation addresses on
   * symmetric heap.
   *
   * When this method completes, the barrier_sync member will be available
   * for use.
   */
  void rocshmem_collective_init();

#ifdef USE_HOST_SIDE_HDP_FLUSH
  /**
   * @brief A service thread routine that flushes the hdp cache on behalf of the
   * GPU.
   */
  void hdp_flush_poll();

  /**
   * @brief Workers used to poll on the device hdp flush request.
   */
  std::thread hdp_flush_worker_thread{};
#endif

  /**
   * @brief Signals to the worker threads to exist
   */
  std::atomic<bool> worker_thread_exit{false};

 public:
  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the GPUIBBackend
   */
  HostInterface *host_interface{nullptr};

  /**
   * @brief Handle for raw memory for barrier sync
   */
  long *barrier_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for reduce sync
   */
  long *reduce_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for broadcast sync
   */
  long *bcast_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall sync
   */
  long *alltoall_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for work
   */
  void *pWrk_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall
   */
  void *pAta_pool{nullptr};

  /**
   * @brief rocSHMEM's copy of MPI_COMM_WORLD (for interoperability
   * with orthogonal MPI usage in an MPI+rocSHMEM program).
   */
  MPI_Comm gpu_ib_comm_world{};
  MPI_Comm backend_comm{};

  /**
   * @brief Holds number of blocks used in library
   */
  size_t num_blocks_{1};

 private:
  /**
   * @brief Allocates cacheable, device memory for the hdp policy.
   *
   * @note Internal data ownership is managed by the proxy
   */
  HdpProxy<HIPAllocator> hdp_proxy_{};

 public:
  /**
   * @brief Policy choice for two HDP implementations.
   *
   * @todo Combine HDP related stuff together into a class with a
   * reasonable interface. The functionality does not need to exist in
   * multiple pieces in the Backend and QueuePair classes. The hdp_rkey,
   * hdp_addresses, and hdp_policy fields should all live in the class.
   */
  HdpPolicy *hdp_policy{hdp_proxy_.get()};

  /**
   * @brief Scratchpad for the internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Compile-time configuration policy for network (IB)
   *
   *
   * The configuration option "USE_SINGLE_NODE" can be enabled to not build
   * with network support.
   */
  NetworkImpl networkImpl{};

 private:
  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  GPUIBContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, GPUIBContext *> ctx_free_list{};

  /**
   * @brief Holds maximum number of contexts used in library
   */
  size_t maximum_num_contexts_{1024};

  /**
   * @brief The bitmask representing the availability of teams in the pool
   */
  char *pool_bitmask_{nullptr};

  /**
   * @brief Bitmask to store the reduced result of bitmasks on pariticipating
   * PEs
   *
   * With no thread-safety for this bitmask, multithreaded creation of teams is
   * not supported.
   */
  char *reduced_bitmask_{nullptr};

  /**
   * @brief Size of the bitmask
   */
  int bitmask_size_{-1};

  /**
   * @brief a helper thread to perform the initialization (non-blocking init)
   */
  std::thread async_thread_{};

  /**
   * @brief Holds a copy of the default context (see OpenSHMEM
   * specification).
   *
   * @todo Remove this member from the backend class. There is another
   * copy stored in ROCSHMEM_CTX_DEFAULT.
   */
  GPUIBContext *default_ctx_{nullptr};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  GPUIBHostContext *default_host_ctx_{nullptr};

  unsigned int* hdp_gpu_cpu_flush_flag_;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP_

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_HIP) && __HIPCC__
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <benchmark/benchmark.h>

#include <Benchmark_Context.hpp>
#include <Benchmark_Utils.hpp>

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosGraph_LoadBalance.hpp"

using namespace KokkosKernelsBenchmark;
using HierarchicalHints = KokkosGraph::Impl::HierarchicalBalancerHints;

template <typename TasksView, typename RanksView, typename TaskEndsView>
void check_correctness(benchmark::State &state, 
  const TasksView &tasks,
  const RanksView &ranks,
  const TaskEndsView &taskEnds,
  const DieOnError &die, const SkipOnError &skip) {


  auto htasks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tasks);
  auto hranks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ranks);
  auto htaskEnds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), taskEnds);

  //TODO

}

struct NativeLoadBalance {

template <typename Task, typename ExecSpace, typename TasksView, typename RanksView, typename TaskEndsView>
  static void lb(const ExecSpace &space,
          const TasksView &tasks,
          const RanksView &ranks,
          const TaskEndsView &taskEnds, typename TaskEndsView::value_type totalWork,
          const HierarchicalHints &hints = HierarchicalHints()) {

    static_assert(std::is_same_v<typename TaskEndsView::non_const_value_type, typename RanksView::non_const_value_type>, "");
    static_assert(std::is_same_v<Task, typename TasksView::non_const_value_type>, "");

    KokkosGraph::Impl::load_balance_inclusive_into<Task>(space, tasks, ranks, taskEnds, totalWork, hints);
  }

  static std::string name() {
    return "native";
  }

};


struct MergeLoadBalance {

  template <typename Task, typename ExecSpace, typename TasksView, typename RanksView, typename TaskEndsView>
  static void lb(const ExecSpace &space,
          const TasksView &tasks,
          const RanksView &ranks,
          const TaskEndsView &taskEnds, typename TaskEndsView::value_type totalWork,
          const HierarchicalHints &/*hints*/) {

    using work_item_type = typename TaskEndsView::non_const_value_type;
    static_assert(std::is_same_v<work_item_type, typename RanksView::non_const_value_type>, "");
    static_assert(std::is_same_v<Task, typename TasksView::non_const_value_type>, "");
  
    auto stepper = KOKKOS_LAMBDA(const KokkosGraph::StepDirection &dir, const KokkosGraph::StepperContext &step,
                      const KokkosGraph::StepperContext &thread) {
      if (KokkosGraph::StepDirection::b == dir) {
        // recover global index into a and b
        Task ag   = thread.ai + step.ai;
        work_item_type bg   = thread.bi + step.bi;
        tasks(bg) = ag;
        ranks(bg) = bg - (ag > 0 ? taskEnds(ag - 1) : 0);
      }
    };

    KokkosKernels::Impl::Iota<work_item_type> b(totalWork);
    KokkosGraph::merge_path(space, taskEnds, b, stepper);
  }

    static std::string name() {
    return "merge-path";
  }

};


template <typename Balancer, typename Task, typename TaskEndsView>
void run(benchmark::State &state,
  const TaskEndsView &taskEnds,
  const typename TaskEndsView::value_type totalWork,
  const HierarchicalHints &hints = HierarchicalHints()) {

  using device_type = typename TaskEndsView::device_type;
  using execution_space = typename device_type::execution_space;
  using work_item_type = typename TaskEndsView::non_const_value_type;

  Kokkos::View<Task *, device_type> tasks("", totalWork);
  Kokkos::View<work_item_type *, device_type> ranks("", totalWork);

  state.counters["tasks"] = taskEnds.size();
  state.counters["work_items"] = totalWork;

  execution_space space;
  Balancer::template lb<Task>(space, tasks, ranks, taskEnds, totalWork, hints);
  Kokkos::fence();
  check_correctness(state, tasks, ranks, taskEnds, DieOnError(false), SkipOnError(true));

  for (auto _ : state) {
    Balancer::template lb<Task>(space, tasks, ranks, taskEnds, totalWork, hints);
    Kokkos::fence();
  }

  const size_t bytesPer = taskEnds.size() * sizeof(work_item_type)  // taskEnds
              + totalWork * sizeof(Task) // tasks output
              + totalWork * sizeof(work_item_type) // ranks output
              ;

  state.SetBytesProcessed(bytesPer * state.iterations());
  state.SetItemsProcessed(taskEnds.size() * state.iterations());
}

template <typename Balancer, typename Task, typename WorkItem, typename Device>
void run_synthetic(benchmark::State &state, const Task n, const HierarchicalHints &hints = HierarchicalHints()) {
  using execution_space = typename Device::execution_space;
  Kokkos::View<WorkItem*, Device> taskEnds("", n);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  fill_random(taskEnds, random_pool, 0, 113);
  KokkosKernels::Impl::kk_inclusive_parallel_prefix_sum<decltype(taskEnds), execution_space>(taskEnds.size(), taskEnds);
  WorkItem totalWork;
  Kokkos::deep_copy(totalWork, Kokkos::subview(taskEnds, taskEnds.size() - 1));
  Kokkos::fence();

  run<Balancer, Task>(state, taskEnds, totalWork, hints);
}

template <typename Balancer, typename Task, typename WorkItem, typename Device>
void run_matrix(benchmark::State &state, const fs::path &path) {
  using dont_care_scalar = float;
  using Matrix = KokkosSparse::CrsMatrix<dont_care_scalar, Task, Device, void, WorkItem>;

  const Matrix crs =
      KokkosSparse::Impl::read_kokkos_crst_matrix<Matrix>(path.c_str());
  using size_type = decltype(crs.graph.row_map.size());
  auto taskEnds = Kokkos::subview(crs.graph.row_map, Kokkos::make_pair(size_type(1), crs.graph.row_map.size()));

  run<Balancer, Task>(state, taskEnds, crs.nnz());
}

template <typename Task, typename WorkItem, typename Device, typename Balancer>
void register_matrix_market(const int argc, const char * const *argv) {
  for (int i = 1; i < argc; ++i) {

    fs::path path(argv[i]);

    std::string name = std::string("MatrixMarket") 
                      + "/" + std::string(path.stem())
                      + "/" + as_string<Task>() 
                      + "/" + as_string<WorkItem>() 
                      + "/" + as_string<Device>() 
                      + "/" + Balancer::name()
                      ;
    benchmark::RegisterBenchmark(name.c_str(), run_matrix<Balancer, Task, WorkItem, Device>, path)->UseRealTime();
  }
}

template <typename Task, typename WorkItem, typename Device>
void register_sweep_params() {

  using TaskEndsView = Kokkos::View<WorkItem*, Device>;
  using Balancer = NativeLoadBalance;

  std::vector<int> teamPathLengths{128, 256, 384, 512, 768, 1024, 1024+128, 1024+256, 1024+512};
  std::vector<int> teamSizes{32, 64, 128, 256, 512, 1024};

  for (auto teamSize : teamSizes) {
    for (auto teamPathLength : teamPathLengths) {
      std::string name = std::string("Params") 
                        + "/" + std::to_string(teamSize)
                        + "/" + std::to_string(teamPathLength)
                      + "/" + as_string<Task>() 
                      + "/" + as_string<WorkItem>() 
                        + "/" + as_string<Device>()
                        + "/" + Balancer::name()
                        ;  
 
      HierarchicalHints hints;
      hints.teamSize = teamSize;
      hints.teamPathLength = teamPathLength;

      benchmark::RegisterBenchmark(name.c_str(), run_synthetic<Balancer, Task, WorkItem, Device>, 10000000, hints)->UseRealTime();
    }
  }
}

template <typename Task, typename WorkItem, typename Device>
void register_sweep_sizes() {

  using TaskEndsView = Kokkos::View<WorkItem*, Device>;
  using Balancer = NativeLoadBalance;

  for (auto n : {100, 1000, 10000, 100000, 1000000, 10000000}) {
    std::string name = std::string("Sizes") 
                     + "/" + std::to_string(n)
                     + "/" + as_string<Task>() 
                     + "/" + as_string<WorkItem>() 
                     + "/" + as_string<Device>()
                     + "/" + Balancer::name()
                     ;  
    benchmark::RegisterBenchmark(name.c_str(), run_synthetic<Balancer, Task, WorkItem, Device>, n, HierarchicalHints())->UseRealTime();
  }
}

template <typename Device>
void register_all(const int argc, const char * const *argv) {
  register_matrix_market<int32_t, int32_t, Device, NativeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, int32_t, Device, MergeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, uint32_t, Device, NativeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, uint32_t, Device, MergeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, int64_t, Device, NativeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, int64_t, Device, MergeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, uint64_t, Device, NativeLoadBalance>(argc, argv);
  register_matrix_market<int32_t, uint64_t, Device, MergeLoadBalance>(argc, argv);

  register_sweep_sizes<int32_t, int32_t, Device>();
  register_sweep_sizes<int32_t, uint32_t, Device>();
  register_sweep_sizes<int32_t, int64_t, Device>();
  register_sweep_sizes<int32_t, uint64_t, Device>();
  register_sweep_sizes<int64_t, int32_t, Device>();
  register_sweep_sizes<int64_t, uint32_t, Device>();
  register_sweep_sizes<int64_t, int64_t, Device>();
  register_sweep_sizes<int64_t, uint64_t, Device>();

  register_sweep_params<int32_t, int32_t, Device>();
  register_sweep_params<int32_t, uint32_t, Device>();
  register_sweep_params<int32_t, int64_t, Device>();
  register_sweep_params<int32_t, uint64_t, Device>();
  register_sweep_params<int64_t, int32_t, Device>();
  register_sweep_params<int64_t, uint32_t, Device>();
  register_sweep_params<int64_t, int64_t, Device>();
  register_sweep_params<int64_t, uint64_t, Device>();

}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMicrosecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);

#if defined(KOKKOS_ENABLE_Serial)
  register_all<Kokkos::Serial>(argc, argv);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  register_all<Kokkos::OpenMP>(argc, argv);
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  register_all<Kokkos::Cuda>(argc, argv);
#endif
#if defined(KOKKOS_ENABLE_HIP)
  register_all<Kokkos::HIP>(argc, argv);
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  register_all<Kokkos::SYCL>(argc, argv);
#endif

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
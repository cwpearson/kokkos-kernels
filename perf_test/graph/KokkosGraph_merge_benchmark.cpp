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

#include <benchmark/benchmark.h>

#include <Benchmark_Context.hpp>
#include <Benchmark_Utils.hpp>
#include <Kokkos_Core.hpp>

#include "Kokkos_Random.hpp"
#include "Kokkos_Sort.hpp"

#include "KokkosGraph_Merge.hpp"
#include "KokkosGraph_MergePath.hpp"

using namespace KokkosKernelsBenchmark;

template <typename View>
void check_correctness(
  benchmark::State &state,
  const View &c, const View &a, const View &b,
  const DieOnError &die, const SkipOnError &skip
) {


  auto on_error = [&](const std::string &msg) {
    if (skip) {state.SkipWithError(msg.c_str()); return; }
    if (die) exit(EXIT_FAILURE);
  };

  if (a.size() + b.size() != c.size()) {
    on_error("size_mismatch");
  }

  
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);

  size_t ia = 0;
  size_t ib = 0;
  size_t ic = 0;
  for(; ia < ha.size() && ib < hb.size() && ic < hc.size(); ) {
    auto av = ha(ia);
    auto bv = hb(ib);

    if (av < bv) {
      if (av != hc(ic)) {
        on_error("wrong");
      }
      ++ia;
    } else {
      if (bv != hc(ic)) {
        on_error("wrong");
      }
      ++ib;
    }
    ++ic;
  }
  if (ia + ib != ic) {
    on_error("didn't consume both input arrays");
  }

}

struct Native {

template <typename ExecSpace, typename View>
  static void merge(const ExecSpace &space,
          const View &c, const View &a, const View &b) {
    KokkosGraph::merge_into(space, c, a, b);
  }

  static std::string name() {
    return "native";
  }

};


struct MergePath {

  template <typename ExecSpace, typename View>
  static void merge(const ExecSpace &space,
          const View &c, const View &a, const View &b) {

 
      auto stepper = KOKKOS_LAMBDA(const KokkosGraph::StepDirection &dir,
                                  const KokkosGraph::StepperContext &step,
                                  const KokkosGraph::StepperContext &thread) {
        if (KokkosGraph::StepDirection::a == dir) {
          c(step.pi + thread.pi) = a(step.ai + thread.ai);
        } else {
          c(step.pi + thread.pi) = b(step.ai + thread.ai);
        }
      };

    KokkosGraph::merge_path(space, a, b, stepper);
  }

    static std::string name() {
    return "merge-path";
  }

};

template <typename View>
void init(benchmark::State &state, View &c, View &a, View &b) {

  using device_type = typename View::device_type;
  using execution_space = typename device_type::execution_space;

  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  fill_random(a, random_pool, 0.0, 100.0);
  fill_random(b, random_pool, 0.0, 100.0);
  Kokkos::sort(a);
  Kokkos::sort(b);
  Kokkos::fence();
}

template <typename Merge, typename T, typename Device>
void run(benchmark::State &state, size_t aSz, size_t bSz) {

  using execution_space = typename Device::execution_space;
  using view_type = Kokkos::View<T *, Device>;

  view_type a("a", aSz);
  view_type b("b", bSz);
  view_type c("c", aSz + bSz);
  init(state, c, a, b);

  state.counters["a_size"] = aSz;
  state.counters["b_size"] = bSz;

  execution_space space;
  Merge::template merge(space, c, a, b);
  Kokkos::fence();
  check_correctness(state, c, a, b, DieOnError(false), SkipOnError(true));

  for (auto _ : state) {
    Merge::template merge(space, c, a, b);
    Kokkos::fence();
  }

  const size_t bytesPer = 2 * (aSz + bSz) * sizeof(T);
  state.SetBytesProcessed(bytesPer * state.iterations());
  state.SetItemsProcessed((aSz + bSz) * state.iterations());
}

template <typename T, typename Device>
void run_native(benchmark::State &state, size_t sz, const KokkosGraph::Impl::MergePathHints &hints) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ARCH_VOLTA)
  if constexpr (std::is_same_v<Device, Kokkos::Cuda>) {
    if (hints.teamSize && hints.threadPathLength && ((*hints.teamSize) * (*hints.threadPathLength) * sizeof(T) * 3 >= 47*1024)) {
      state.SkipWithError("too large for Volta?");
      return;
    }
  }
#endif

  using execution_space = typename Device::execution_space;
  using view_type = Kokkos::View<T *, Device>;
  using M = KokkosGraph::Impl::HierarchicalMerger<execution_space, view_type, view_type, view_type>;

  view_type a("a", sz);
  view_type b("b", sz);
  view_type c("c", sz + sz);
  init(state, c, a, b);

  state.counters["a_size"] = sz;
  state.counters["b_size"] = sz;

  execution_space space;
  M::launch(space, c, a, b, hints);
  Kokkos::fence();
  check_correctness(state, c, a, b, DieOnError(false), SkipOnError(true));

  for (auto _ : state) {
    M::launch(space, c, a, b, hints);
    Kokkos::fence();
  }

  const size_t bytesPer = 2 * (sz + sz) * sizeof(T);
  state.SetBytesProcessed(bytesPer * state.iterations());
  state.SetItemsProcessed((sz + sz) * state.iterations());
}

template <typename Merge, typename T, typename Device>
void register_sweep_sizes() {

  for (auto a : {100, 1000, 10000, 100000, 1000000, 10000000}) {
    for (auto b : {100, 1000, 10000, 100000, 1000000, 10000000}) {
      std::string name = std::string("Sizes") 
                      + "/" + as_string<T>() 
                      + "/" + as_string<Device>()
                      + "/" + Merge::name()
                      + "/" + std::to_string(a)
                      + "/" + std::to_string(b)
                      ;  
      benchmark::RegisterBenchmark(name.c_str(), run<Merge, T, Device>, a, b)->UseRealTime();
    }
  }
}

template <typename T, typename Device>
void register_sweep_params() {

  for (auto teamSize : {32, 64, 128, 256, 512}) {
    for (auto threadPathLength : {1, 2, 3, 4, 6,7, 8, 10, 12, 16}) {

      std::string name = std::string("Params") 
                      + "/" + as_string<T>() 
                      + "/" + as_string<Device>()
                      + "/native"
                      + "/" + std::to_string(teamSize)
                      + "/" + std::to_string(threadPathLength)
                      ;  

      KokkosGraph::Impl::MergePathHints hints;
      hints.threadPathLength = threadPathLength;
      hints.teamSize = teamSize;

      benchmark::RegisterBenchmark(name.c_str(), run_native<T, Device>, 10000000, hints)->UseRealTime();

    }
  }
}

template <typename Device>
void register_all() {
  register_sweep_sizes<Native, int32_t, Device>();
  register_sweep_sizes<Native, uint32_t, Device>();
  register_sweep_sizes<Native, int64_t, Device>();
  register_sweep_sizes<Native, uint64_t, Device>();
  register_sweep_sizes<MergePath, int32_t, Device>();
  register_sweep_sizes<MergePath, uint32_t, Device>();
  register_sweep_sizes<MergePath, int64_t, Device>();
  register_sweep_sizes<MergePath, uint64_t, Device>();
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMicrosecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);

#if defined(KOKKOS_ENABLE_Serial)
  register_all<Kokkos::Serial>();
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  register_all<Kokkos::OpenMP>();
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  register_all<Kokkos::Cuda>();
  register_sweep_params<int32_t, Kokkos::Cuda>();
  register_sweep_params<uint32_t, Kokkos::Cuda>();
  register_sweep_params<int64_t, Kokkos::Cuda>();
  register_sweep_params<uint64_t, Kokkos::Cuda>();
#endif
#if defined(KOKKOS_ENABLE_HIP)
  register_all<Kokkos::HIP>();
  register_sweep_params<int32_t, Kokkos::HIP>();
  register_sweep_params<uint32_t, Kokkos::HIP>();
  register_sweep_params<int64_t, Kokkos::HIP>();
  register_sweep_params<uint64_t, Kokkos::HIP>();
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  register_all<Kokkos::SYCL>();
  register_sweep_params<int32_t, Kokkos::SYCL>();
  register_sweep_params<uint32_t, Kokkos::SYCL>();
  register_sweep_params<int64_t, Kokkos::SYCL>();
  register_sweep_params<uint64_t, Kokkos::SYCL>();
#endif

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
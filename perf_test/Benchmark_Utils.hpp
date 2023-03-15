/*
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
*/

#ifndef KOKKOSKERNELS_PERFTEST_BENCHMARK_UTILS_HPP
#define KOKKOSKERNELS_PERFTEST_BENCHMARK_UTILS_HPP

namespace KokkosKernelsBenchmark {

template <typename...> inline constexpr bool always_false = false;

template <typename T>
std::string as_string() {
  // On mac,
  // uint64_t = unsigned long long
  // uint32_t = unsigned
  // so we have to have an unsigned long test

  if constexpr (false) { // formatting consistency
  } else if constexpr(std::is_same_v<T, uint64_t>) {
    return "u64";
  } else if constexpr(std::is_same_v<T, uint32_t>) {
    return "u32";
  } else if constexpr(std::is_same_v<T, unsigned long>) {
    return "ul";
  } else if constexpr(std::is_same_v<T, uint32_t>) {
    return "u32";
  } else if constexpr(std::is_same_v<T, int64_t>) {
    return "i64";
  } else if constexpr(std::is_same_v<T, int32_t>) {
    return "i32";
  } else if constexpr(std::is_same_v<T, float>) {
    return "f32";
  } else if constexpr(std::is_same_v<T, double>) {
    return "f64";
  } 
#if defined(KOKKOS_ENABLE_CUDA)
  else if constexpr(std::is_same_v<T, Kokkos::Cuda>) {
    return "CUDA";
  }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  else if constexpr(std::is_same_v<T, Kokkos::OpenMP>) {
    return "OpenMP";
  }
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
  else if constexpr(std::is_same_v<T, Kokkos::Serial>) {
    return "Serial";
  }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  else if constexpr(std::is_same_v<T, Kokkos::SYCL>) {
    return "SYCL";
  }
#endif
#if defined(KOKKOS_ENABLE_HIP)
  else if constexpr(std::is_same_v<T, Kokkos::HIP>) {
    return "HIP";
  }
#endif
  else {
    static_assert(always_false<T>, "unhandled type for as_string");
  }
}

class WrappedBool {
 public:
  WrappedBool(const bool &val) : val_(val) {}

  operator bool() const { return val_; }

 protected:
  bool val_;
};

class DieOnError : public WrappedBool {
 public:
  DieOnError(const bool &val) : WrappedBool(val) {}
};
class SkipOnError : public WrappedBool {
 public:
  SkipOnError(const bool &val) : WrappedBool(val) {}
};

}  // namespace KokkosKernelsBenchmark

#endif  // KOKKOSKERNELS_PERFTEST_BENCHMARK_UTILS_HPP
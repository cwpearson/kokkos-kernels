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

#include "KokkosKernels_Unreachable.hpp"

namespace KokkosKernelsBenchmark {

// for use in static asserts
template <typename...>
inline constexpr bool always_false = false;

template <typename T>
std::string as_string() {
  // On mac,
  // uint64_t = unsigned long long
  // uint32_t = unsigned
  // so we have to have an unsigned long test

  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::is_signed_v<T>) {
      if constexpr (sizeof(T) == 1) {
        return "i8";
      } else if constexpr (sizeof(T) == 2) {
        return "i16";
      } else if constexpr (sizeof(T) == 4) {
        return "i32";
      } else if constexpr (sizeof(T) == 8) {
        return "i64";
      } else if constexpr (sizeof(T) == 16) {
        return "i128";
      } else {
        static_assert(always_false<T>, "unexpected signed integral size");
      }
    } else {
      if constexpr (sizeof(T) == 1) {
        return "u8";
      } else if constexpr (sizeof(T) == 2) {
        return "u16";
      } else if constexpr (sizeof(T) == 4) {
        return "u32";
      } else if constexpr (sizeof(T) == 8) {
        return "u64";
      } else if constexpr (sizeof(T) == 16) {
        return "u128";
      } else {
        static_assert(always_false<T>, "unexpected unsigned integral size");
      }
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    if constexpr (sizeof(T) == 1) {
      return "f8";
    } else if constexpr (sizeof(T) == 2) {
      return "f16";
    } else if constexpr (sizeof(T) == 4) {
      return "f32";
    } else if constexpr (sizeof(T) == 8) {
      return "f64";
    } else if constexpr (sizeof(T) == 16) {
      return "f128";
    } else {
      static_assert(always_false<T>, "unexpected float size");
    }    
  }
#if defined(KOKKOS_ENABLE_CUDA)
  else if constexpr (std::is_same_v<T, Kokkos::Cuda>) {
    return "CUDA";
  }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  else if constexpr (std::is_same_v<T, Kokkos::OpenMP>) {
    return "OpenMP";
  }
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
  else if constexpr (std::is_same_v<T, Kokkos::Serial>) {
    return "Serial";
  }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
  else if constexpr (std::is_same_v<T, Kokkos::SYCL>) {
    return "SYCL";
  }
#endif
#if defined(KOKKOS_ENABLE_HIP)
  else if constexpr (std::is_same_v<T, Kokkos::HIP>) {
    return "HIP";
  }
#endif
  else {
    static_assert(always_false<T>, "unhandled type for as_string");
  }
  KOKKOSKERNELS_IMPL_UNREACHABLE();
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
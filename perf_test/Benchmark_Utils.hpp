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

template <typename T>
std::string as_string();

template <>
std::string as_string<uint64_t>() {
  return "u64";
}

template <>
std::string as_string<unsigned long>() {
  switch (sizeof(unsigned long)) {
    case 4: return "u32";
    case 8: return "u64";
    default: return "u??";
  }
}

template <>
std::string as_string<uint32_t>() {
  return "u32";
}

template <>
std::string as_string<int64_t>() {
  return "i64";
}

template <>
std::string as_string<int32_t>() {
  return "i32";
}

template <>
std::string as_string<float>() {
  return "f32";
}

template <>
std::string as_string<double>() {
  return "f64";
}

#if defined(KOKKOS_ENABLE_CUDA)
template <>
std::string as_string<Kokkos::Cuda>() {
  return "CUDA";
}
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
std::string as_string<Kokkos::OpenMP>() {
  return "OpenMP";
}
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
template <>
std::string as_string<Kokkos::Serial>() {
  return "Serial";
}
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
std::string as_string<Kokkos::SYCL>() {
  return "SYCL";
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
std::string as_string<Kokkos::HIP>() {
  return "HIP";
}
#endif

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
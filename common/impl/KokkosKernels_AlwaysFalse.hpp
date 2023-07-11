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

#ifndef KOKKOSKERNELS_VIEWUTILS_HPP
#define KOKKOSKERNELS_VIEWUTILS_HPP
#include "Kokkos_Core.hpp"

namespace KokkosKernels::Impl {

// for use in static asserts
template <typename...>
inline constexpr bool always_false = false;

}  // namespace KokkosKernels::Impl

#endif

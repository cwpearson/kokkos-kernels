#ifndef KOKKOSKERNELS_UNREACHABLE_HPP
#define KOKKOSKERNELS_UNREACHABLE_HPP

#include "Kokkos_Macros.hpp"

#if defined(KOKKOS_COMPILER_IBM) || defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG) || defined(__INTEL_COMPILER)

#define KOKKOSKERNELS_IMPL_UNREACHABLE() __builtin_unreachable()

#elif KOKKOS_COMPILER_NVCC >= 1120

#define KOKKOSKERNELS_IMPL_UNREACHABLE() __builtin_assume(false)

#elif defined(KOKKOS_COMPILER_MSVC)

#define KOKKOSKERNELS_IMPL_UNREACHABLE() __assume(false)

#endif // different compilers

#endif // KOKKOSKERNELS_UNREACHABLE_HPP
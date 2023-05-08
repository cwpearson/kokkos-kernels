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

/*! \file

    Taken from
    - Trilinos/packages/tpetra/core/src/Tpetra_BlockCrsMatrix_def.hpp
    - Trilinos/packages/tpetra/core/src/Tpetra_BlockView.hpp
    - Trilinos/packages/tpetra/core/src/Tpetra_BlockCrsMatrix_decl.hpp
 */

#ifndef KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SERIAL_HPP
#define KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SERIAL_HPP

#include <Kokkos_Core.hpp>

#include "KokkosKernels_ViewUtils.hpp"

namespace KokkosSparse {
namespace Impl {

template <unsigned M, unsigned N, typename Alpha, typename AS, typename XS, typename YS>
void gemv_tile(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t aExtent0
  ) {
  YS acc[M] = {0};
  XS xl[N];

  // load x into registers
  #pragma unroll
  for (unsigned i = 0; i < N; ++i) {
    xl[i] = x[i];
  }

  #pragma unroll
  for (unsigned i = 0; i < M; ++i) {
    # pragma unroll
    for (unsigned j = 0; j < N; ++j) {
      acc[i] += a[i * aExtent0 + j] * xl[j];
    }
  }

  // write registers back to memory
  #pragma unroll
  for (unsigned i = 0; i < M; ++i) {
    y[i] += alpha * acc[i];
  }
}

template <unsigned M, typename Alpha, typename AS, typename XS, typename YS>
void gemv_rows(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t rowLen
  ) {

  size_t j = 0;
  for (; j + 7 <= rowLen; j += 7) {
    gemv_tile<M, 7>(alpha, &a[j], &x[j], y, rowLen);
  }
  for (; j + 4 <= rowLen; j += 4) {
    gemv_tile<M, 4>(alpha, &a[j], &x[j], y, rowLen);
  }

  #pragma unroll
  for (size_t i = 0; i < M; ++i) {
    YS acc = 0;
    for (size_t jj = j; jj < rowLen; ++jj) {
      acc += a[i * rowLen + jj] * x[jj];
    }
    y[i] += alpha * acc;
  }

  // for (; j + 2 <= rowLen; j += 2) {
  //   gemv_tile<M, 2>(alpha, &a[j], &x[j], y, rowLen);
  // }
  // for (; j + 1 <= rowLen; j += 1) {
  //   gemv_tile<M, 1>(alpha, &a[j], &x[j], y, rowLen);
  // }
}

template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_mn_tiled(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {

#if 0


  size_t i = 0;
  for (; i < blockSize; i += M) {
    size_t j = 0;
    for (; j + N <= blockSize; j += N) {
        gemv_tile<M,N>(alpha, &a[i * blockSize + j], &x[j], &y[i], blockSize);
    }
    for (; j < blockSize; j += 1) {
        gemv_tile<M,1>(alpha, &a[i * blockSize + j], &x[j], &y[i], blockSize);
    }
  }
#elif 1
  size_t i = 0;
  for (; i + 7 <= blockSize; i += 7) {
    gemv_rows<7>(alpha, &a[i * blockSize], x, &y[i], blockSize);
  }
  for (; i + 4 <= blockSize; i += 4) {
    gemv_rows<4>(alpha, &a[i * blockSize], x, &y[i], blockSize);
  }
  for (; i < blockSize; ++i) {
    auto acc = y[i];
    for (size_t j = 0; j < blockSize; ++j) {
      acc += alpha * a[i * blockSize + j] * x[j];
    }
    y[i] = acc;
  }
#else
  for (size_t i = 0; i < blockSize; ++i) {
    auto acc = y[i];
    for (size_t j = 0; j < blockSize; ++j) {
      acc += alpha * a[i * blockSize + j] * x[j];
    }
    y[i] = acc;
  }
#endif
}

template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_mn_blocked(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {

#if 0


  size_t i = 0;
  for (; i < blockSize; i += M) {
    size_t j = 0;
    for (; j + N <= blockSize; j += N) {
        gemv_tile<M,N>(alpha, &a[i * blockSize + j], &x[j], &y[i], blockSize);
    }
    for (; j < blockSize; j += 1) {
        gemv_tile<M,1>(alpha, &a[i * blockSize + j], &x[j], &y[i], blockSize);
    }
  }
#elif 1
  size_t i = 0;
  for (; i + 8 <= blockSize; i += 8) {
    gemv_rows<8>(alpha, &a[i * blockSize], x, &y[i], blockSize);
  }
  for (; i + 4 <= blockSize; i += 4) {
    gemv_rows<4>(alpha, &a[i * blockSize], x, &y[i], blockSize);
  }
  for (; i < blockSize; ++i) {
    // std::cerr << __FILE__ << ":" << __LINE__ << " i=" << i << "\n";
    auto acc = y[i];
    for (size_t j = 0; j < blockSize; ++j) {
      acc += alpha * a[i * blockSize + j] * x[j];
    }
    y[i] = acc;
  }
#else
  for (size_t i = 0; i < blockSize; ++i) {
    auto acc = y[i];
    for (size_t j = 0; j < blockSize; ++j) {
      acc += alpha * a[i * blockSize + j] * x[j];
    }
    y[i] = acc;
  }
#endif
}


/*! produce a single entry of Y by dotting a row of A with X

*/
template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_n_blocked(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {

    for (size_t i = 0; i < blockSize; ++i) {
      YS acc = 0;
      #pragma unroll(4)
      for (size_t j = 0; j < blockSize; ++j) {
        acc += a[i * blockSize + j] * x[j];
      }
      y[i] += alpha * acc;
    }
}

/*! dot MB rows of `a` with `x`, producing MB products
   `a` is LayoutRight
*/
template <unsigned MB, typename Alpha, typename AS, typename XS, typename YS>
void multidot_unsafe_nostride(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t n
  ) {
    YS yv[MB] = {0};
    for (size_t j = 0; j < n; ++j) {
      const auto xj = x[j];
      #pragma unroll(MB)
      for (unsigned m = 0; m < MB; ++m) {
        yv[m] += a[m * n + j] * xj;
      }
    }
    #pragma unroll(MB)
    for (unsigned m = 0; m < MB; ++m) {
      y[m] += alpha * yv[m];
    }
}

/*! dot MB rows of `a` with `x`, writing MB products into Y

  If the strides can be known to be 1 at compile-time, performance can really be improved. Force inline to try to propagate as much stride info as possible.
*/
template <
unsigned MB, 
typename Alpha, typename AScalar, typename XScalar, typename YScalar
>
KOKKOS_FORCEINLINE_FUNCTION
void multidot_unsafe(
  const Alpha alpha,
  const AScalar * KOKKOS_RESTRICT a,
  const size_t aStride0, // stride between entries in dim 0 of a
  const XScalar * KOKKOS_RESTRICT x,
  YScalar * KOKKOS_RESTRICT y, // at least MB long
  const size_t n // length of rows of a (and x)
  ) {
    YScalar yv[MB] = {0};
    for (size_t j = 0; j < n; ++j) {
      const XScalar xj = x[j];
      #pragma unroll(MB)
      for (unsigned m = 0; m < MB; ++m) {
        yv[m] += a[m * aStride0 + j] * xj;
      }
    }
    #pragma unroll(MB)
    for (unsigned m = 0; m < MB; ++m) {
      y[m] += alpha * yv[m];
    }
}

/*! compute multiple entries of Y at the same time
*/
template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_m_blocked_unsafe_nostride(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {

    size_t i = 0;
    for (; i + 8 <= blockSize; i += 8) {
      multidot_unsafe_nostride<8>(alpha, &a[i * blockSize], x, &y[i], blockSize);
    }
    for (; i + 7 <= blockSize; i += 7) {
      multidot_unsafe_nostride<7>(alpha, &a[i * blockSize], x, &y[i], blockSize);
    }
    for (; i + 4 <= blockSize; i += 4) {
      multidot_unsafe_nostride<4>(alpha, &a[i * blockSize], x, &y[i], blockSize);
    }
    for (; i + 2 <= blockSize; i += 2) {
      multidot_unsafe_nostride<2>(alpha, &a[i * blockSize], x, &y[i], blockSize);
    }
    for (; i + 1 <= blockSize; i += 1) {
      multidot_unsafe_nostride<1>(alpha, &a[i * blockSize], x, &y[i], blockSize);
    }
}

/*! compute multiple entries of Y at the same time
*/
template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_m_blocked_unsafe(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const size_t aStride0,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {

    size_t i = 0;
    for (; i + 8 <= blockSize; i += 8) {
      multidot_unsafe<8>(alpha, &a[i * blockSize], aStride0, x, &y[i], blockSize);
    }
    for (; i + 5 <= blockSize; i += 5) {
      multidot_unsafe<5>(alpha, &a[i * blockSize], aStride0, x, &y[i], blockSize);
    }
    for (; i + 4 <= blockSize; i += 4) {
      multidot_unsafe<4>(alpha, &a[i * blockSize], aStride0, x, &y[i], blockSize);
    }
    for (; i + 2 <= blockSize; i += 2) {
      multidot_unsafe<2>(alpha, &a[i * blockSize], aStride0, x, &y[i], blockSize);
    }
    for (; i + 1 <= blockSize; i += 1) {
      multidot_unsafe<1>(alpha, &a[i * blockSize], aStride0, x, &y[i], blockSize);
    }
}

/*! dot MB rows of `a` with `x`, producing MB products
   y[0:MB] += A[0:MB , :] * x
*/
template <unsigned MB, typename Alpha, typename AMatrix, typename XVector, typename YVector>
void multidot(
  const Alpha alpha,
  const AMatrix &a,
  const XVector &x,
  const YVector &y
  ) {

    static_assert(AMatrix::rank == 2, "");
    static_assert(XVector::rank == 1, "");
    static_assert(YVector::rank == 1, "");

    static_assert(std::is_same_v<typename AMatrix::array_layout, Kokkos::LayoutRight>, "");
    static_assert(!std::is_same_v<typename XVector::array_layout, Kokkos::LayoutStride>, "");
    static_assert(!std::is_same_v<typename YVector::array_layout, Kokkos::LayoutStride>, "");

    multidot_unsafe<MB>(
      alpha,
      a.data(), a.stride_0(),
      x.data(),
      y.data(),
      x.size()
    );

}

/*! compute multiple entries of Y at the same time

    require x and y to be contiguous
*/
template <typename Alpha, typename AMatrix, typename XVector, typename YVector>
void gemv_m_blocked(
  const Alpha alpha,
  const AMatrix &a,
  const XVector &x,
  const YVector &y
  ) {
    
    const size_t numRows = y.size();
    size_t i = 0;
    for (; i + 8 <= numRows; i += 8) {
      const Kokkos::pair rows{size_t(i), size_t(i+8)};
      auto as = Kokkos::subview(a, rows, Kokkos::ALL);
      auto ys = Kokkos::subview(y, rows);
      multidot<8>(alpha, as, x, ys);
    }
    for (; i + 7 <= numRows; i += 7) {
      const Kokkos::pair rows{size_t(i), size_t(i+7)};
      auto as = Kokkos::subview(a, rows, Kokkos::ALL);
      auto ys = Kokkos::subview(y, rows);
      multidot<7>(alpha, as, x, ys);
    }
    for (; i + 4 <= numRows; i += 4) {
      const Kokkos::pair rows{size_t(i), size_t(i+4)};
      auto as = Kokkos::subview(a, rows, Kokkos::ALL);
      auto ys = Kokkos::subview(y, rows);
      multidot<4>(alpha, as, x, ys);
    }
    for (; i + 2 <= numRows; i += 2) {
      const Kokkos::pair rows{size_t(i), size_t(i+2)};
      auto as = Kokkos::subview(a, rows, Kokkos::ALL);
      auto ys = Kokkos::subview(y, rows);
      multidot<2>(alpha, as, x, ys);
    }
    for (; i + 1 <= numRows; i += 1) {
      const Kokkos::pair rows{size_t(i), size_t(i+1)};
      auto as = Kokkos::subview(a, rows, Kokkos::ALL);
      auto ys = Kokkos::subview(y, rows);
      multidot<1>(alpha, as, x, ys);
    }
}

/*! produce a single entry of Y by dotting a row of A with X

*/
template <typename Alpha, typename AS, typename XS, typename YS>
void gemv_n_blocked_2(
  const Alpha alpha,
  const AS * KOKKOS_RESTRICT a,
  const XS * KOKKOS_RESTRICT x,
  YS * KOKKOS_RESTRICT y,
  const size_t blockSize
  ) {


    constexpr size_t NB = 8;

    for (size_t i = 0; i < blockSize; ++i) {
      size_t j = 0;

      // vector-size chunks
      YS yv[NB] = {0};
      YS acc = 0;
      for (; j + NB <= blockSize; j += NB) {
        auto aij = &a[i * blockSize + j];
        auto xj = &x[j];
        #pragma clang loop vectorize_width(NB) interleave_count(NB)
        for (size_t jj = 0; jj < NB; ++jj) {
          yv[jj] += aij[jj] * xj[jj];
        }
      }
      #pragma unroll(NB)
      for (size_t jj = 0; jj < NB; ++jj) {
        acc += yv[jj];
      }   
      for (; j < blockSize; ++j) {
        acc += a[i * blockSize + j] * x[j];
      }
   
      y[i] += alpha * acc;
    }
}

template <typename Alpha, typename AMatrix, typename XVector, typename Beta, typename YVector>
void bsr_spmv(const Alpha alpha, 
const AMatrix &a, 
const XVector &x, const Beta beta, const YVector &y) {
  using a_ordinal_type = typename AMatrix::non_const_ordinal_type;
  static_assert(XVector::rank == 1, "");
  static_assert(YVector::rank == 1, "");


  const auto * KOKKOS_RESTRICT aRows = a.graph.row_map.data();
  const auto * KOKKOS_RESTRICT aCols = a.graph.entries.data();
  const a_ordinal_type blockSize = a.blockDim();
  const a_ordinal_type numRows = a.numRows();

  // TODO:
  // optionally pack X and Y into contiguous versions if they're non-contiguous

  auto * KOKKOS_RESTRICT yp = y.data();
  const auto * KOKKOS_RESTRICT aVals = a.values.data();

  // scale y
  for (size_t i = 0; i < numRows * blockSize; ++i) {
    y[i] = beta * y[i];
  }

#if 0
  // cheaper subviews
  auto ux = KokkosKernels::Impl::make_unmanaged(x);
  auto uy = KokkosKernels::Impl::make_unmanaged(y);
#endif

  for (a_ordinal_type i = 0; i < numRows; ++i) {
    const size_t rowBegin = aRows[i];
    const size_t rowEnd = aRows[i+1];

    for (size_t ji = rowBegin; ji < rowEnd; ++ji) {
      const a_ordinal_type j = aCols[ji];

#if 0
      gemv_mn_blocked(alpha,
           &aVals[ji * blockSize * blockSize], 
           &x[j * blockSize],
           &y[i * blockSize],
           blockSize);
#elif 0

      gemv_n_blocked_2(alpha,
           &aVals[ji * blockSize * blockSize], 
           &x[j * blockSize],
           &y[i * blockSize],
           blockSize);
#elif 0
      const Kokkos::pair blockCols{size_t(j*blockSize), size_t((j+1)*blockSize)};
      const Kokkos::pair blockRows{size_t(i*blockSize), size_t((i+1)*blockSize)};
      auto xs = Kokkos::subview(ux, blockCols);
      auto ys = Kokkos::subview(uy, blockRows);
      // std::cerr << __FILE__ << ":" << __LINE__ 
      //           << " gemv on block" << i << "," << j << " (" << ji << ")\n";
      gemv_m_blocked(alpha,
           a.unmanaged_block(ji),
           xs,
           ys);
#elif 0
      gemv_m_blocked_unsafe_nostride(alpha,
           &aVals[ji * blockSize * blockSize], 
           &x[j * blockSize],
           &y[i * blockSize],
           blockSize);
#elif 1
      gemv_m_blocked_unsafe(alpha,
           &aVals[ji * blockSize * blockSize],
           blockSize,
           &x[j * blockSize],
           &y[i * blockSize],
           blockSize);
#endif
    }
  }

  // Kokkos::deep_copy(y, yll);
}

// This was not provided
template <typename Alpha, typename AMatrix, typename XVector, typename Beta,
          typename YVector>
void apply_serial(const Alpha &alpha, const AMatrix &a, const XVector &x,
                  const Beta &beta, const YVector &y) {

  using x_value_type = typename XVector::non_const_value_type;
  using y_value_type = typename YVector::non_const_value_type;

  Kokkos::RangePolicy<typename YVector::execution_space> policy(0, y.size());
  if constexpr(YVector::rank == 1 && XVector::rank == 1) {
    bsr_spmv(alpha, a, x, beta, y);
  } else {
    // apply to each vector in turn (just to check correctness)
    for (size_t k = 0; k < y.extent(1); ++k) {
      auto ys = Kokkos::subview(y, Kokkos::ALL, k);
      auto xs = Kokkos::subview(x, Kokkos::ALL, k);
      apply_serial(alpha, a, xs, beta, ys);
    }
  }
} // apply_serial


}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SERIAL_HPP

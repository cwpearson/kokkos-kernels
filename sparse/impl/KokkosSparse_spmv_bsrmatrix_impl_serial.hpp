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
  for (size_t i = 0; i < M; ++i) {
    # pragma unroll
    for (size_t j = 0; j < N; ++j) {
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
  for (; j + 8 <= rowLen; j += 8) {
    gemv_tile<M, 8>(alpha, &a[j], &x[j], y, rowLen);
  }
  for (; j + 4 <= rowLen; j += 4) {
    gemv_tile<M, 4>(alpha, &a[j], &x[j], y, rowLen);
  }

  // TODO: something wrong with this
  // for (size_t i = 0; i < M; ++i) {
  //   auto acc = 0;
  //   for (size_t jj = j; jj < rowLen; ++jj) {
  //     acc += a[i * rowLen + jj] * x[jj];
  //   }
  //   y[i] += alpha * acc;
  // }

  for (; j + 2 <= rowLen; j += 2) {
    gemv_tile<M, 2>(alpha, &a[j], &x[j], y, rowLen);
  }
  for (; j + 1 <= rowLen; j += 1) {
    gemv_tile<M, 1>(alpha, &a[j], &x[j], y, rowLen);
  }
}

template <typename Alpha, typename AS, typename XS, typename YS>
void gemv(
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



template <typename Alpha, typename AS, typename AR, typename AC, typename XS, typename Beta, typename YS>
void bsr_spmv(const Alpha &alpha, 
const AS * KOKKOS_RESTRICT aVals, 
const AR * KOKKOS_RESTRICT aRows, 
const AC *KOKKOS_RESTRICT aCols, 
const XS * KOKKOS_RESTRICT x, const Beta beta, YS * KOKKOS_RESTRICT y,
const size_t numRows, const size_t blockSize) {

  // scale y
  for (size_t i = 0; i < numRows * blockSize; ++i) {
    y[i] = beta * y[i];
  }

  for (size_t i = 0; i < numRows; ++i) {
    const size_t rowBegin = aRows[i];
    const size_t rowEnd = aRows[i+1];

    for (size_t ji = rowBegin; ji < rowEnd; ++ji) {
      const size_t j = aCols[ji];

      // std::cerr << __FILE__ << ":" << __LINE__ << " " << i << "," << j << "\n";

      gemv(alpha,
           &aVals[ji * blockSize * blockSize], 
           &x[j * blockSize],
           &y[i * blockSize],
           blockSize);

    }
  }

}

// This was not provided
template <typename Alpha, typename AMatrix, typename XVector, typename Beta,
          typename YVector>
void apply_serial(const Alpha &alpha, const AMatrix &a, const XVector &x,
                  const Beta &beta, const YVector &y) {

  using x_value_type = typename XVector::non_const_value_type;
  using y_value_type = typename YVector::non_const_value_type;


  Kokkos::RangePolicy<typename YVector::execution_space> policy(0, y.size());
  if constexpr(YVector::rank == 1) {

    const auto * KOKKOS_RESTRICT aVals = a.values.data();
    const auto * KOKKOS_RESTRICT aRows = a.graph.row_map.data();
    const auto * KOKKOS_RESTRICT aCols = a.graph.entries.data();
    const auto * KOKKOS_RESTRICT xp = x.data();
    y_value_type * KOKKOS_RESTRICT yp = y.data();


    bsr_spmv(alpha, aVals, aRows, aCols, xp, beta, yp, a.numRows(), a.blockDim());


  } else {

    // get a single vector
    auto y1 = Kokkos::subview(y, Kokkos::ALL, 0);
    return apply_serial(alpha, a, x, beta, y1);

  }
} // apply_serial


}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SERIAL_HPP

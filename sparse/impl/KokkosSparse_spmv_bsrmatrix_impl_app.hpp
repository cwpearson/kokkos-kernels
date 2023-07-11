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

#ifndef KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_APP_HPP
#define KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_APP_HPP

#include <Kokkos_Core.hpp>

#include <KokkosKernels_ViewUtils.hpp>

namespace KokkosSparse {
namespace Impl {

// Siefert did not provide these params, so this is a guess
template <typename RowMap, typename Entries, typename BlockValues,
          typename ConstMVView, typename MVView>
// actually LinearSolverBelosBlock::MatrixOperatorGPU::MvpKernel
class AppKernel {
  // these typedefs were not provided
  using LO   = typename Entries::non_const_value_type;
  using Real = typename BlockValues::non_const_value_type;

  const RowMap A_rp_;
  const Entries A_ci_;
  const BlockValues A_d_;
  ConstMVView x_;
  MVView y_;
  LO nrhs_, nblockrows_, blocksz_, bsz2_;

 public:
  AppKernel(const RowMap &A_rp, const Entries &A_ci, const BlockValues &A_d,
              ConstMVView x, MVView y)
      : A_rp_(A_rp),
        A_ci_(A_ci),
        A_d_(A_d),
        x_(x),
        y_(y)
        /*adding these */,
        nrhs_(x_.extent(1)),
        nblockrows_(A_rp_.extent(0) - 1),
        blocksz_(y_.extent(0) / nblockrows_),
        bsz2_(blocksz_ * blocksz_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const size_t k) const {
    const LO irhs     = k / y_.extent(0);
    const LO row      = k % y_.extent(0);
    const LO blockrow = row / blocksz_;
    const LO lclrow   = row % blocksz_;
    Real accum        = 0;
    const LO j_lim    = A_rp_(blockrow + 1);
    for (LO j = A_rp_(blockrow); j < j_lim; ++j) {
#ifdef TBCRS_TRANSPOSE_BLOCK
      typedef Kokkos::LayoutLeft BlockLayout;
#else
      typedef Kokkos::LayoutRight BlockLayout;
#endif
      typedef Kokkos::View<const Real **, BlockLayout,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          ConstBlock;
      ConstBlock b(&A_d_(bsz2_ * j), blocksz_, blocksz_);
      const LO blockcol = A_ci_(j);
      const LO x_start  = blockcol * blocksz_;
      const auto x_lcl  = Kokkos::subview(
          x_, Kokkos::make_pair(x_start, x_start + blocksz_), irhs);
      for (LO i = 0; i < blocksz_; ++i) accum += b(lclrow, i) * x_lcl(i);
    }
    y_(row, irhs) = accum;
  }

  void run() { Kokkos::parallel_for(y_.size(), *this); }
};

// This was not provided
template <typename Alpha, typename AMatrix, typename XVector, typename Beta,
          typename YVector>
void apply_app(const Alpha &alpha, const AMatrix &a, const XVector &x,
                 const Beta &beta, const YVector &y) {

  Kokkos::RangePolicy<typename YVector::execution_space> policy(0, y.size());
  if constexpr(YVector::rank == 1) {
    const Kokkos::View<typename YVector::value_type*[1], typename YVector::device_type, typename YVector::memory_traits> yu(y.data(), y.extent(0), 1);
    const Kokkos::View<typename XVector::value_type*[1], typename XVector::device_type, typename XVector::memory_traits> xu(x.data(), x.extent(0), 1);
    AppKernel op(a.graph.row_map, a.graph.entries, a.values, xu, yu);
    Kokkos::parallel_for(policy, op);
  } else {
    AppKernel op(a.graph.row_map, a.graph.entries, a.values, x, y);
    Kokkos::parallel_for(policy, op);
  }
}


/* One thread for each entry in the product multivector

   Each thread accumulates the partial products for its entry, and writes it out.
*/
template <
typename Alpha,
typename AMatrix,
typename XVector,
typename Beta,
typename YVector>
class NonTransRangeYEntries {

  using LO = typename AMatrix::non_const_ordinal_type;

  Alpha alpha_;
  AMatrix a_;
  XVector x_;
  Beta beta_;
  YVector y_;

 public:
  NonTransRangeYEntries(const Alpha &alpha, const AMatrix &a, const XVector &x, const Beta &beta, const YVector &y)
      : alpha_(alpha),
        a_(a),
        x_(x),
        beta_(beta),
        y_(y) {}

  template <unsigned BLOCK_SIZE = 0>
  KOKKOS_INLINE_FUNCTION void impl(const size_t k) const {

    using a_value_type = typename AMatrix::non_const_value_type;
    using Accum = typename YVector::non_const_value_type;
    using BlockLayout = Kokkos::LayoutRight;
    using ConstBlock = Kokkos::View<const Accum **, BlockLayout,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const LO irhs = k / y_.extent(0);
    const LO row  = k % y_.extent(0);

    // scale by beta
    if (0 == beta_) {
      y_(row, irhs) = 0; // convert NaN to 0
    } else if (1 != beta_) {
      y_(row, irhs) *= beta_;
    }

    // for non-zero template instantiations,
    // constant propagation should optimize divmod
    LO blocksz;
    if constexpr(0 == BLOCK_SIZE) {
      blocksz = a_.blockDim();
    } else {
      blocksz = BLOCK_SIZE;
    }

    if (0 != alpha_) {
      const LO bsz2 = blocksz * blocksz;
      const LO blockRow = row / blocksz;
      const LO lclrow   = row % blocksz;
      Accum accum       = 0;
      const LO j_begin = a_.graph.row_map(blockRow);
      const LO j_end   = a_.graph.row_map(blockRow + 1);
      for (LO j = j_begin; j < j_end; ++j) {
        ConstBlock b(&a_.values(bsz2 * j), blocksz, blocksz);
        const LO blockcol = a_.graph.entries(j);
        const LO x_start  = blockcol * blocksz;
        const auto x_lcl  = Kokkos::subview(
            x_, Kokkos::make_pair(x_start, x_start + blocksz), irhs);
        for (LO i = 0; i < blocksz; ++i) accum += b(lclrow, i) * x_lcl(i);
      }
      y_(row, irhs) = alpha_ * accum;
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const size_t k) const {

    if (false) {}
    // clang-format off
    else if ( 1 == a_.blockDim()) { impl<1>(k); }
    else if ( 2 == a_.blockDim()) { impl<2>(k); }
    else if ( 3 == a_.blockDim()) { impl<3>(k); }
    else if ( 4 == a_.blockDim()) { impl<4>(k); }
    else if ( 5 == a_.blockDim()) { impl<5>(k); }
    else if ( 6 == a_.blockDim()) { impl<6>(k); }
    else if ( 7 == a_.blockDim()) { impl<7>(k); }
    else if ( 8 == a_.blockDim()) { impl<8>(k); }
    else if ( 9 == a_.blockDim()) { impl<9>(k); }
    else if (10 == a_.blockDim()) { impl<10>(k); }
    else if (11 == a_.blockDim()) { impl<11>(k); }
    // clang-format on
    else {impl<0>(k);}
  }


};

template <typename Alpha, typename AMatrix, typename XVector, typename Beta,
          typename YVector>
void apply_modified_app(const Alpha &alpha, const AMatrix &a, const XVector &x,
                 const Beta &beta, const YVector &y) {

  using execution_space = typename YVector::execution_space;

  Kokkos::RangePolicy<execution_space> policy(0, y.size());
  if constexpr(YVector::rank == 1) {

    // convert to a 2D view with extent 1 in the second dimension
    using Y2D = KokkosKernels::Impl::with_unmanaged_t<Kokkos::View<typename YVector::value_type*[1], typename YVector::device_type, typename YVector::memory_traits>>;
    using X2D = KokkosKernels::Impl::with_unmanaged_t<Kokkos::View<typename XVector::value_type*[1], typename XVector::device_type, typename XVector::memory_traits>>;
    const Y2D yu(y.data(), y.extent(0), 1);
    const X2D xu(x.data(), x.extent(0), 1);
    NonTransRangeYEntries op(alpha, a, xu, beta, yu);
    Kokkos::parallel_for(policy, op);
  } else {
    NonTransRangeYEntries op(alpha, a, x, beta, y);
    Kokkos::parallel_for(policy, op);
  }
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_APP_HPP

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

#ifndef KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SPARC_HPP
#define KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SPARC_HPP

#include <Kokkos_Core.hpp>

namespace KokkosSparse {
namespace Impl {

// Siefert did not provide these params, so this is a guess
template <typename RowMap, typename Entries, typename BlockValues,
          typename ConstMVView, typename MVView>
// actually LinearSolverBelosBlock::MatrixOperatorGPU::MvpKernel
class SparcKernel {
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
  SparcKernel(const RowMap &A_rp, const Entries &A_ci, const BlockValues &A_d,
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
void apply_sparc(const Alpha &alpha, const AMatrix &a, const XVector &x,
                 const Beta &beta, const YVector &y) {

  Kokkos::RangePolicy<typename YVector::execution_space> policy(0, y.size());
  if constexpr(YVector::rank == 1) {
    const Kokkos::View<typename YVector::value_type*[1], typename YVector::device_type, typename YVector::memory_traits> yu(y.data(), y.extent(0), 1);
    const Kokkos::View<typename XVector::value_type*[1], typename XVector::device_type, typename XVector::memory_traits> xu(x.data(), x.extent(0), 1);
    SparcKernel op(a.graph.row_map, a.graph.entries, a.values, xu, yu);
    Kokkos::parallel_for(policy, op);
  } else {
    SparcKernel op(a.graph.row_map, a.graph.entries, a.values, x, y);
    Kokkos::parallel_for(policy, op);
  }


}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_SPARC_HPP

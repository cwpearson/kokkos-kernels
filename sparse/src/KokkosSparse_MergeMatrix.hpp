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

#ifndef KOKKOSSPARSE_MERGEMATRIX_HPP
#define KOKKOSSPARSE_MERGEMATRIX_HPP

#include <type_traits>

#include "KokkosKernels_Iota.hpp"
#include "KokkosKernels_SafeCompare.hpp"

/// \file KokkosSparse_MergeMatrix.hpp

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

/*! \class MergeMatrixDiagonal
    \brief a view into the entries of the Merge Matrix along a diagonal

   @tparam AView Type of the input view a, must be rank 1
   @tparam BViewLike Type of the view-like object b, must be Kokkos::View or
  KokkosKernels::Iota Example merge matrix M of two arrays A (vertical) and B
  (horizontal), as seen in Odeh, Green, Mwassi, Shmueli, Birk Merge Path -
  Parallel Merging Made Simple 2012 M[i,j] = 1 iff A[i] > B[j] operator(k)
  returns A[i] > B[j] at the kth entry of the diagonal

        3  5 12 22 45 64 69 82
      ------------------------
      |  /           /
   17 | 1  1  1  0  0  0  0  0
      |/          /
   29 | 1  1  1  1  0  0  0  0
      |         /
   35 | 1  1  1  1  0  0  0  0
      |     /
   73 | 1  1  1  1  1  1  1  0
      |   /
   86 | 1  1  1  1  1  1  1  1
      |/
   90 | 1  1  1  1  1  1  1  1
      |
   95 | 1  1  1  1  1  1  1  1
      |
   99 | 1  1  1  1  1  1  1  1
  Diagonals are counted from the top-left.
  Index into a diagonal from the bottom-left.
  Shown on the figure above is the 1st and 5th diagonal
  The 0th diagonal D_0 has length 0
  The 1st diagonal D_1 has length 1
  The 5th diagonal D_5 has length 5
  The 9th diagonal D_9 has length 7
  D_1(0) = 1
  D_5(0..3) = 1
  D_5(4) = 0
*/
template <typename AView, typename BViewLike>
class MergeMatrixDiagonal {
 public:
  static_assert(AView::rank == 1, "MergeMatrixDiagonal AView must be rank 1");
  static_assert(Kokkos::is_view_v<BViewLike> ||
                    KokkosKernels::Impl::is_iota_v<BViewLike>,
                "MergeMatrixDiagonal BViewLike must be Kokkos::View or "
                "KokkosKernels::Iota");
  static_assert(BViewLike::rank == 1,
                "MergeMatrixDiagonal BViewLike must be rank 1");

  using execution_space = typename AView::execution_space;

  /**
   * Define the types for index and value of each view
   */
  using a_index_type = typename AView::size_type;
  using b_index_type = typename BViewLike::size_type;
  using a_value_type = typename AView::non_const_value_type;
  using b_value_type = typename BViewLike::non_const_value_type;

  /*! \struct MatrixPosition
   *  \brief indices into the a_ and b_ views.
   */
  struct MatrixPosition {
    a_index_type ai;
    b_index_type bi;
  };
  using position_type = MatrixPosition;

  // implement bare minimum parts of the view interface
  enum { rank = 1 };
  using non_const_value_type = bool;  ///< Merge matrix entries are 0 or 1.

  using size_type =
      typename std::conditional<sizeof(typename AView::size_type) >=
                                    sizeof(typename BViewLike::size_type),
                                typename AView::size_type,
                                typename BViewLike::size_type>::
          type;  ///< The larger of the two view types' size_types

  /** \brief Initializes the view a and view-like object b and the diagonal.
   */
  KOKKOS_INLINE_FUNCTION
  MergeMatrixDiagonal(const AView &a, const BViewLike &b,
                      const size_type diagonal)
      : a_(a), b_(b), d_(diagonal) {}
  MergeMatrixDiagonal() = default;

  /**
   * Computes the position along a and b for a given diagonal di
   *
   * @param di Current diagonal
   * @return The MatrixPosition corresponding to the current diagonal
   */
  KOKKOS_INLINE_FUNCTION
  position_type position(const size_type &di) const noexcept {
    position_type pos;
    if (0 == d_) {
      pos.ai = 0;
      pos.bi = 0;
      return pos;
    } else {
      pos = diag_to_a_b(di);
      pos.ai += 1;
      return pos;
    }
  }

  /**
   * Compares a[i] > b[j] along the diagonal at entry di
   *
   * @param di Current diagonal
   * @return True if a[i] > b[j], false otherwise
   */
  KOKKOS_INLINE_FUNCTION
  bool operator()(const size_type di) const {
    position_type pos = diag_to_a_b(di);
    if (pos.ai >= a_.size()) {
      return true;  // on the +a side out of matrix bounds is 1
    } else if (pos.bi >= b_.size()) {
      return false;  // on the +b side out of matrix bounds is 0
    } else {
      return KokkosKernels::Impl::safe_gt(a_(pos.ai), b_(pos.bi));
    }
  }

  /**
   * Returns the length of the diagonal
   *
   * @return Length of the diagonal
   */
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    if (d_ <= a_.size() && d_ <= b_.size()) {
      return d_;
    } else if (d_ > a_.size() && d_ > b_.size()) {
      // TODO: this returns nonsense if d_ happens to be outside the merge
      // matrix
      return a_.size() + b_.size() - d_;
    } else {
      return KOKKOSKERNELS_MACRO_MIN(a_.size(), b_.size());
    }
  }

 private:
  /**
   * Translates an index along the diagonal to indices into a_ and b_
   *
   * @param di Current diagonal
   * @return The corresponding MatrixPosition with indices into a_ and b_
   */
  KOKKOS_INLINE_FUNCTION
  position_type diag_to_a_b(const size_type &di) const noexcept {
    position_type res;
    res.ai = d_ < a_.size() ? (d_ - 1) - di : a_.size() - 1 - di;
    res.bi = d_ < a_.size() ? di : d_ + di - a_.size();
    return res;
  }

  AView a_;      ///< The a view
  BViewLike b_;  ///< The b view
  size_type d_;  ///< diagonal
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_MERGEMATRIX_HPP

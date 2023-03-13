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

#ifndef KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_TPETRA_HPP
#define KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_TPETRA_HPP

#include <Kokkos_Core.hpp>

#include "KokkosBlas1_scal.hpp"

namespace KokkosSparse {
namespace Impl {

#if defined(TPETRA_ENABLE_BLOCKCRS_LITTLEBLOCK_LAYOUTLEFT)
using BlockCrsMatrixLittleBlockArrayLayout = Kokkos::LayoutLeft;
#else
using BlockCrsMatrixLittleBlockArrayLayout = Kokkos::LayoutRight;
#endif

namespace TpetraImpl {

template <class VecType1, class BlkType, class VecType2, class CoeffType,
          class IndexType = int, bool is_contiguous = false,
          class BlkLayoutType = typename BlkType::array_layout>
struct GEMV {
  static KOKKOS_INLINE_FUNCTION void run(const CoeffType& alpha,
                                         const BlkType& A, const VecType1& x,
                                         const VecType2& y) {
    static_assert(VecType1::rank == 1, "GEMV: VecType1 must have rank 1.");
    static_assert(BlkType::rank == 2, "GEMV: BlkType must have rank 2.");
    static_assert(VecType2::rank == 1, "GEMV: VecType2 must have rank 1.");

    const IndexType numRows = static_cast<IndexType>(A.extent(0));
    const IndexType numCols = static_cast<IndexType>(A.extent(1));

    /// general case
    for (IndexType i = 0; i < numRows; ++i)
      for (IndexType j = 0; j < numCols; ++j) y(i) += alpha * A(i, j) * x(j);
  }
};

template <class VecType1, class BlkType, class VecType2, class CoeffType,
          class IndexType>
struct GEMV<VecType1, BlkType, VecType2, CoeffType, IndexType, true,
            Kokkos::LayoutLeft> {
  static KOKKOS_INLINE_FUNCTION void run(const CoeffType& alpha,
                                         const BlkType& A, const VecType1& x,
                                         const VecType2& y) {
    static_assert(VecType1::rank == 1, "GEMV: VecType1 must have rank 1.");
    static_assert(BlkType::rank == 2, "GEMV: BlkType must have rank 2.");
    static_assert(VecType2::rank == 1, "GEMV: VecType2 must have rank 1.");

    using A_value_type = typename std::decay<decltype(A(0, 0))>::type;
    using x_value_type = typename std::decay<decltype(x(0))>::type;
    using y_value_type = typename std::decay<decltype(y(0))>::type;

    const IndexType numRows = static_cast<IndexType>(A.extent(0));
    const IndexType numCols = static_cast<IndexType>(A.extent(1));

    const A_value_type* __restrict__ A_ptr(A.data());
    const IndexType as1(A.stride(1));
    const x_value_type* __restrict__ x_ptr(x.data());
    y_value_type* __restrict__ y_ptr(y.data());

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (IndexType j = 0; j < numCols; ++j) {
      const x_value_type x_at_j               = alpha * x_ptr[j];
      const A_value_type* __restrict__ A_at_j = A_ptr + j * as1;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (IndexType i = 0; i < numRows; ++i) y_ptr[i] += A_at_j[i] * x_at_j;
    }
  }
};

template <class VecType1, class BlkType, class VecType2, class CoeffType,
          class IndexType>
struct GEMV<VecType1, BlkType, VecType2, CoeffType, IndexType, true,
            Kokkos::LayoutRight> {
  static KOKKOS_INLINE_FUNCTION void run(const CoeffType& alpha,
                                         const BlkType& A, const VecType1& x,
                                         const VecType2& y) {
    static_assert(VecType1::rank == 1, "GEMV: VecType1 must have rank 1.");
    static_assert(BlkType::rank == 2, "GEMV: BlkType must have rank 2.");
    static_assert(VecType2::rank == 1, "GEMV: VecType2 must have rank 1.");

    using A_value_type = typename std::decay<decltype(A(0, 0))>::type;
    using x_value_type = typename std::decay<decltype(x(0))>::type;
    using y_value_type = typename std::decay<decltype(y(0))>::type;

    const IndexType numRows = static_cast<IndexType>(A.extent(0));
    const IndexType numCols = static_cast<IndexType>(A.extent(1));

    const A_value_type* __restrict__ A_ptr(A.data());
    const IndexType as0(A.stride(0));
    const x_value_type* __restrict__ x_ptr(x.data());
    y_value_type* __restrict__ y_ptr(y.data());

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (IndexType i = 0; i < numRows; ++i) {
      y_value_type y_at_i(0);
      const auto A_at_i = A_ptr + i * as0;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (IndexType j = 0; j < numCols; ++j) y_at_i += A_at_i[j] * x_ptr[j];
      y_ptr[i] += alpha * y_at_i;
    }
  }
};

}  // namespace TpetraImpl

/// \brief <tt>y := y + alpha * A * x</tt> (dense matrix-vector multiply)
///
/// \param alpha [in] Coefficient by which to multiply A*x (this does
///   NOT necessarily follow BLAS rules; the caller is responsible for
///   checking whether alpha == 0 and implementing BLAS rules in that
///   case).
/// \param A [in] Small dense matrix (must have rank 2)
/// \param x [in] Small dense vector input (must have rank 1 and at
///   least as many rows as A has columns)
/// \param y [in/out] Small dense vector output (must have rank 1 and
///   at least as many rows as A has rows)
template <class VecType1, class BlkType, class VecType2, class CoeffType,
          class IndexType = int>
KOKKOS_INLINE_FUNCTION void GEMV(const CoeffType& alpha, const BlkType& A,
                                 const VecType1& x, const VecType2& y) {
  constexpr bool is_A_contiguous = (std::is_same<typename BlkType::array_layout,
                                                 Kokkos::LayoutLeft>::value ||
                                    std::is_same<typename BlkType::array_layout,
                                                 Kokkos::LayoutRight>::value);
  constexpr bool is_x_contiguous =
      (std::is_same<typename VecType1::array_layout,
                    Kokkos::LayoutLeft>::value ||
       std::is_same<typename VecType1::array_layout,
                    Kokkos::LayoutRight>::value);
  constexpr bool is_y_contiguous =
      (std::is_same<typename VecType2::array_layout,
                    Kokkos::LayoutLeft>::value ||
       std::is_same<typename VecType2::array_layout,
                    Kokkos::LayoutRight>::value);
  constexpr bool is_contiguous =
      is_A_contiguous && is_x_contiguous && is_y_contiguous;
  TpetraImpl::GEMV<VecType1, BlkType, VecType2, CoeffType, IndexType,
                   is_contiguous>::run(alpha, A, x, y);
}

namespace TpetraImpl {

/// \brief Implementation of Tpetra::SCAL function.
///
/// This is the "generic" version that we don't implement.
/// We actually implement versions for ViewType rank 1 or rank 2.
template <class ViewType, class CoefficientType, class IndexType = int,
          const bool is_contiguous = false, const int rank = ViewType::rank>
struct SCAL {
  static KOKKOS_INLINE_FUNCTION void run(const CoefficientType& alpha,
                                         const ViewType& x);
};

/// \brief Implementation of Tpetra::SCAL function, for
///   ViewType rank 1 (i.e., a vector).
template <class ViewType, class CoefficientType, class IndexType>
struct SCAL<ViewType, CoefficientType, IndexType, false, 1> {
  /// \brief x := alpha*x (rank-1 x, i.e., a vector)
  static KOKKOS_INLINE_FUNCTION void run(const CoefficientType& alpha,
                                         const ViewType& x) {
    const IndexType numRows = static_cast<IndexType>(x.extent(0));

    /// general case
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (IndexType i = 0; i < numRows; ++i) x(i) = alpha * x(i);
  }
};
/// \brief Implementation of Tpetra::SCAL function, for
///   ViewType rank 2 (i.e., a matrix).
template <class ViewType, class CoefficientType, class IndexType>
struct SCAL<ViewType, CoefficientType, IndexType, false, 2> {
  /// \brief A := alpha*A (rank-2 A, i.e., a matrix)
  static KOKKOS_INLINE_FUNCTION void run(const CoefficientType& alpha,
                                         const ViewType& A) {
    const IndexType numRows = static_cast<IndexType>(A.extent(0));
    const IndexType numCols = static_cast<IndexType>(A.extent(1));

    for (IndexType j = 0; j < numCols; ++j)
      for (IndexType i = 0; i < numRows; ++i) A(i, j) = alpha * A(i, j);
  }
};
template <class ViewType, class CoefficientType, class IndexType,
          const int rank>
struct SCAL<ViewType, CoefficientType, IndexType, true, rank> {
  /// \brief x := alpha*x (rank-1 x, i.e., a vector)
  static KOKKOS_INLINE_FUNCTION void run(const CoefficientType& alpha,
                                         const ViewType& x) {
    using x_value_type   = typename std::decay<decltype(*x.data())>::type;
    const IndexType span = static_cast<IndexType>(x.span());
    x_value_type* __restrict__ x_ptr(x.data());
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (IndexType i = 0; i < span; ++i) x_ptr[i] = alpha * x_ptr[i];
  }
};

}  // namespace TpetraImpl

/// \brief x := alpha*x, where x is either rank 1 (a vector) or rank 2
///   (a matrix).
template <class ViewType, class CoefficientType, class IndexType = int,
          const int rank = ViewType::rank>
KOKKOS_INLINE_FUNCTION void SCAL(const CoefficientType& alpha,
                                 const ViewType& x) {
  using LayoutType = typename ViewType::array_layout;
  constexpr bool is_contiguous =
      (std::is_same<LayoutType, Kokkos::LayoutLeft>::value ||
       std::is_same<LayoutType, Kokkos::LayoutRight>::value);
  TpetraImpl::SCAL<ViewType, CoefficientType, IndexType, is_contiguous,
                   rank>::run(alpha, x);
}

/// \brief Implementation of Tpetra::FILL function.
///
/// This is the "generic" version that we don't implement.
/// We actually implement versions for ViewType rank 1 or rank 2.
template <class ViewType, class InputType, class IndexType = int,
          const bool is_contiguous = false, const int rank = ViewType::rank>
struct FILL {
  static KOKKOS_INLINE_FUNCTION void run(const ViewType& x,
                                         const InputType& val);
};

/// \brief Implementation of Tpetra::FILL function, for
///   ViewType rank 1 (i.e., a vector).
template <class ViewType, class InputType, class IndexType>
struct FILL<ViewType, InputType, IndexType, false, 1> {
  static KOKKOS_INLINE_FUNCTION void run(const ViewType& x,
                                         const InputType& val) {
    const IndexType numRows = static_cast<IndexType>(x.extent(0));
    for (IndexType i = 0; i < numRows; ++i) x(i) = val;
  }
};
/// \brief Implementation of Tpetra::FILL function, for
///   ViewType rank 2 (i.e., a matrix).
template <class ViewType, class InputType, class IndexType>
struct FILL<ViewType, InputType, IndexType, false, 2> {
  static KOKKOS_INLINE_FUNCTION void run(const ViewType& X,
                                         const InputType& val) {
    const IndexType numRows = static_cast<IndexType>(X.extent(0));
    const IndexType numCols = static_cast<IndexType>(X.extent(1));
    for (IndexType j = 0; j < numCols; ++j)
      for (IndexType i = 0; i < numRows; ++i) X(i, j) = val;
  }
};
template <class ViewType, class InputType, class IndexType, const int rank>
struct FILL<ViewType, InputType, IndexType, true, rank> {
  static KOKKOS_INLINE_FUNCTION void run(const ViewType& x,
                                         const InputType& val) {
    const IndexType span = static_cast<IndexType>(x.span());
    auto x_ptr           = x.data();
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
    for (IndexType i = 0; i < span; ++i) x_ptr[i] = val;
  }
};

template <typename T>
struct BlockCrsRowStruct {
  T totalNumEntries, totalNumBytes, maxRowLength;
  KOKKOS_DEFAULTED_FUNCTION BlockCrsRowStruct() = default;
  KOKKOS_DEFAULTED_FUNCTION BlockCrsRowStruct(const BlockCrsRowStruct& b) =
      default;
  KOKKOS_INLINE_FUNCTION BlockCrsRowStruct(const T& numEnt, const T& numBytes,
                                           const T& rowLength)
      : totalNumEntries(numEnt),
        totalNumBytes(numBytes),
        maxRowLength(rowLength) {}
};  // BlockCrsRowStruct

template <typename T>
static KOKKOS_INLINE_FUNCTION void operator+=(BlockCrsRowStruct<T>& a,
                                              const BlockCrsRowStruct<T>& b) {
  a.totalNumEntries += b.totalNumEntries;
  a.totalNumBytes += b.totalNumBytes;
  a.maxRowLength =
      a.maxRowLength > b.maxRowLength ? a.maxRowLength : b.maxRowLength;
}

template <typename T, typename ExecSpace>
struct BlockCrsReducer {
  typedef BlockCrsReducer reducer;
  typedef T value_type;
  typedef Kokkos::View<value_type, ExecSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      result_view_type;
  value_type* value;

  KOKKOS_INLINE_FUNCTION
  BlockCrsReducer(value_type& val) : value(&val) {}

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, value_type& src) const {
    dst += src;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type& dst,
                                   const value_type& src) const {
    dst += src;
  }
  KOKKOS_INLINE_FUNCTION void init(value_type& val) const {
    val = value_type();
  }
  KOKKOS_INLINE_FUNCTION value_type& reference() { return *value; }
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(value);
  }
};  // BlockCrsReducer

template <class AlphaCoeffType, class GraphType, class MatrixValuesType,
          class InVecType, class BetaCoeffType, class OutVecType,
          bool IsBuiltInType>
class BcrsApplyNoTransFunctor {
 private:
  static_assert(Kokkos::is_view<MatrixValuesType>::value,
                "MatrixValuesType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<OutVecType>::value,
                "OutVecType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<InVecType>::value,
                "InVecType must be a Kokkos::View.");
  static_assert(std::is_same<MatrixValuesType,
                             typename MatrixValuesType::const_type>::value,
                "MatrixValuesType must be a const Kokkos::View.");
  static_assert(std::is_same<typename OutVecType::value_type,
                             typename OutVecType::non_const_value_type>::value,
                "OutVecType must be a nonconst Kokkos::View.");
  static_assert(std::is_same<typename InVecType::value_type,
                             typename InVecType::const_value_type>::value,
                "InVecType must be a const Kokkos::View.");
  static_assert(static_cast<int>(MatrixValuesType::rank) == 1,
                "MatrixValuesType must be a rank-1 Kokkos::View.");
  static_assert(static_cast<int>(InVecType::rank) == 1,
                "InVecType must be a rank-1 Kokkos::View.");
  static_assert(static_cast<int>(OutVecType::rank) == 1,
                "OutVecType must be a rank-1 Kokkos::View.");
  using scalar_type = typename MatrixValuesType::non_const_value_type;

 public:
  using device_type = typename GraphType::device_type;

  //! Type of the (mesh) column indices in the sparse graph / matrix.
  using local_ordinal_type =
      typename std::remove_const<typename GraphType::data_type>::type;

  /// \brief Set the current vector / current column of the input
  ///   (multi)vector X to use.
  ///
  /// This lets us handle multiple columns by iterating over them one
  /// column at a time, without needing to recreate the functor each
  /// time.
  void setX(const InVecType& X) { X_ = X; }

  /// \brief Set the current vector / current column of the output
  ///   (multi)vector Y to use.
  ///
  /// This lets us handle multiple columns by iterating over them one
  /// column at a time, without needing to recreate the functor each
  /// time.
  void setY(const OutVecType& Y) { Y_ = Y; }

  typedef typename Kokkos::ArithTraits<
      typename std::decay<AlphaCoeffType>::type>::val_type alpha_coeff_type;
  typedef typename Kokkos::ArithTraits<
      typename std::decay<BetaCoeffType>::type>::val_type beta_coeff_type;

  //! Constructor.
  BcrsApplyNoTransFunctor(const alpha_coeff_type& alpha, const GraphType& graph,
                          const MatrixValuesType& val,
                          const local_ordinal_type blockSize,
                          const InVecType& X, const beta_coeff_type& beta,
                          const OutVecType& Y)
      : alpha_(alpha),
        ptr_(graph.row_map),
        ind_(graph.entries),
        val_(val),
        blockSize_(blockSize),
        X_(X),
        beta_(beta),
        Y_(Y) {}

#if 0
    // Dummy team version
    KOKKOS_INLINE_FUNCTION void
    operator () (const typename Kokkos::TeamPolicy<typename device_type::execution_space>::member_type & member) const
    {
      Kokkos::abort("Tpetra::BcrsApplyNoTransFunctor:: this should not be called");
    }
#endif

  // Range Policy for non built-in types
  KOKKOS_INLINE_FUNCTION void operator()(
      const local_ordinal_type& lclRow) const {
    using Kokkos::Details::ArithTraits;
    using KAT = Kokkos::Details::ArithTraits<beta_coeff_type>;

    typedef typename decltype(ptr_)::non_const_value_type offset_type;
    typedef Kokkos::View<typename MatrixValuesType::const_value_type**,
                         BlockCrsMatrixLittleBlockArrayLayout, device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        little_block_type;

    const offset_type Y_ptBeg = lclRow * blockSize_;
    const offset_type Y_ptEnd = Y_ptBeg + blockSize_;
    auto Y_cur = Kokkos::subview(Y_, Kokkos::make_pair(Y_ptBeg, Y_ptEnd));

    // This version of the code does not use temporary storage.
    // Each thread writes to its own block of the target vector.
    if (beta_ == ArithTraits<beta_coeff_type>::zero()) {
      KokkosBlas::fill(Y_cur, ArithTraits<beta_coeff_type>::zero());
    } else if (beta_ !=
               ArithTraits<beta_coeff_type>::one()) {  // beta != 0 && beta != 1
      KokkosBlas::scal(Y_cur, beta_, Y_cur);
    }

    if (alpha_ != ArithTraits<alpha_coeff_type>::zero()) {
      const offset_type blkBeg = ptr_[lclRow];
      const offset_type blkEnd = ptr_[lclRow + 1];
      // Precompute to save integer math in the inner loop.
      const offset_type bs2 = blockSize_ * blockSize_;
      for (offset_type absBlkOff = blkBeg; absBlkOff < blkEnd; ++absBlkOff) {
        little_block_type A_cur(val_.data() + absBlkOff * bs2, blockSize_,
                                blockSize_);
        const offset_type X_blkCol = ind_[absBlkOff];
        const offset_type X_ptBeg  = X_blkCol * blockSize_;
        const offset_type X_ptEnd  = X_ptBeg + blockSize_;
        auto X_cur = subview(X_, ::Kokkos::make_pair(X_ptBeg, X_ptEnd));

        GEMV(alpha_, A_cur, X_cur, Y_cur);  // Y_cur += alpha*A_cur*X_cur
      }  // for each entry in current local block row of matrix
    }
  }

 private:
  alpha_coeff_type alpha_;
  typename GraphType::row_map_type::const_type ptr_;
  typename GraphType::entries_type::const_type ind_;
  MatrixValuesType val_;
  local_ordinal_type blockSize_;
  InVecType X_;
  beta_coeff_type beta_;
  OutVecType Y_;
};  // BcrsApplyNoTransFunctor

template <class AlphaCoeffType, class GraphType, class MatrixValuesType,
          class InVecType, class BetaCoeffType, class OutVecType>
class BcrsApplyNoTransFunctor<AlphaCoeffType, GraphType, MatrixValuesType,
                              InVecType, BetaCoeffType, OutVecType, true> {
 private:
  static_assert(Kokkos::is_view<MatrixValuesType>::value,
                "MatrixValuesType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<OutVecType>::value,
                "OutVecType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<InVecType>::value,
                "InVecType must be a Kokkos::View.");
  static_assert(std::is_same<MatrixValuesType,
                             typename MatrixValuesType::const_type>::value,
                "MatrixValuesType must be a const Kokkos::View.");
  static_assert(std::is_same<typename OutVecType::value_type,
                             typename OutVecType::non_const_value_type>::value,
                "OutVecType must be a nonconst Kokkos::View.");
  static_assert(std::is_same<typename InVecType::value_type,
                             typename InVecType::const_value_type>::value,
                "InVecType must be a const Kokkos::View.");
  static_assert(static_cast<int>(MatrixValuesType::rank) == 1,
                "MatrixValuesType must be a rank-1 Kokkos::View.");
  static_assert(static_cast<int>(InVecType::rank) == 1,
                "InVecType must be a rank-1 Kokkos::View.");
  static_assert(static_cast<int>(OutVecType::rank) == 1,
                "OutVecType must be a rank-1 Kokkos::View.");
  typedef typename MatrixValuesType::non_const_value_type scalar_type;

 public:
  typedef typename GraphType::device_type device_type;

  //! Does this Functor get run on the host or on the device
  static constexpr bool runOnHost =
      !std::is_same_v<typename device_type::execution_space,
                      Kokkos::DefaultExecutionSpace> ||
      std::is_same_v<Kokkos::DefaultExecutionSpace,
                     Kokkos::DefaultHostExecutionSpace>;

  //! Type of the (mesh) column indices in the sparse graph / matrix.
  typedef typename std::remove_const<typename GraphType::data_type>::type
      local_ordinal_type;

  /// \brief Set the current vector / current column of the input
  ///   (multi)vector X to use.
  ///
  /// This lets us handle multiple columns by iterating over them one
  /// column at a time, without needing to recreate the functor each
  /// time.
  void setX(const InVecType& X) { X_ = X; }

  /// \brief Set the current vector / current column of the output
  ///   (multi)vector Y to use.
  ///
  /// This lets us handle multiple columns by iterating over them one
  /// column at a time, without needing to recreate the functor each
  /// time.
  void setY(const OutVecType& Y) { Y_ = Y; }

  typedef typename Kokkos::ArithTraits<
      typename std::decay<AlphaCoeffType>::type>::val_type alpha_coeff_type;
  typedef typename Kokkos::ArithTraits<
      typename std::decay<BetaCoeffType>::type>::val_type beta_coeff_type;

  //! Constructor.
  BcrsApplyNoTransFunctor(const alpha_coeff_type& alpha, const GraphType& graph,
                          const MatrixValuesType& val,
                          const local_ordinal_type blockSize,
                          const InVecType& X, const beta_coeff_type& beta,
                          const OutVecType& Y)
      : alpha_(alpha),
        ptr_(graph.row_map),
        ind_(graph.entries),
        val_(val),
        blockSize_(blockSize),
        X_(X),
        beta_(beta),
        Y_(Y) {}

  // dummy Range version
  KOKKOS_INLINE_FUNCTION void operator()(
      const local_ordinal_type& lclRow) const {
    Kokkos::abort(
        "Tpetra::BcrsApplyNoTransFunctor:: this should not be called");
  }

  // Team policy for built-in types
  KOKKOS_INLINE_FUNCTION void operator()(
      const typename Kokkos::TeamPolicy<
          typename device_type::execution_space>::member_type& member) const {
    const local_ordinal_type lclRow = member.league_rank();

    using Kokkos::Details::ArithTraits;
    // I'm not writing 'using Kokkos::make_pair;' here, because that
    // may break builds for users who make the mistake of putting
    // 'using namespace std;' in the global namespace.  Please don't
    // ever do that!  But just in case you do, I'll take this
    // precaution.
    using Kokkos::parallel_for;
    using Kokkos::subview;
    typedef typename decltype(ptr_)::non_const_value_type offset_type;
    typedef Kokkos::View<typename MatrixValuesType::const_value_type**,
                         BlockCrsMatrixLittleBlockArrayLayout, device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        little_block_type;

    const offset_type Y_ptBeg = lclRow * blockSize_;
    const offset_type Y_ptEnd = Y_ptBeg + blockSize_;
    auto Y_cur = subview(Y_, ::Kokkos::make_pair(Y_ptBeg, Y_ptEnd));

    // This version of the code does not use temporary storage.
    // Each thread writes to its own block of the target vector.
    if (beta_ == ArithTraits<beta_coeff_type>::zero()) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blockSize_),
                           [&](const local_ordinal_type& i) {
                             Y_cur(i) = ArithTraits<beta_coeff_type>::zero();
                           });
    } else if (beta_ !=
               ArithTraits<beta_coeff_type>::one()) {  // beta != 0 && beta != 1
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, blockSize_),
          [&](const local_ordinal_type& i) { Y_cur(i) *= beta_; });
    }
    member.team_barrier();

    if (alpha_ != ArithTraits<alpha_coeff_type>::zero()) {
      const offset_type blkBeg = ptr_[lclRow];
      const offset_type blkEnd = ptr_[lclRow + 1];
      // Precompute to save integer math in the inner loop.
      const offset_type bs2 = blockSize_ * blockSize_;
      little_block_type A_cur(val_.data(), blockSize_, blockSize_);
      auto X_cur = subview(X_, ::Kokkos::make_pair(0, blockSize_));
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, blkBeg, blkEnd),
          [&](const local_ordinal_type& absBlkOff) {
            A_cur.assign_data(val_.data() + absBlkOff * bs2);
            const offset_type X_blkCol = ind_[absBlkOff];
            const offset_type X_ptBeg  = X_blkCol * blockSize_;
            X_cur.assign_data(&X_(X_ptBeg));

            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, blockSize_),
                [&](const local_ordinal_type& k0) {
                  scalar_type val(0);
                  for (local_ordinal_type k1 = 0; k1 < blockSize_; ++k1)
                    val += A_cur(k0, k1) * X_cur(k1);
                  if constexpr (runOnHost) {
                    // host space team size is always 1
                    Y_cur(k0) += alpha_ * val;
                  } else {
                    // cuda space team size can be larger than 1
                    // atomic is not allowed for sacado type;
                    // thus this needs to be specialized or
                    // sacado atomic should be supported.
                    Kokkos::atomic_add(&Y_cur(k0), alpha_ * val);
                  }
                });
          });  // for each entry in current local block row of matrix
    }
  }

 private:
  alpha_coeff_type alpha_;
  typename GraphType::row_map_type::const_type ptr_;
  typename GraphType::entries_type::const_type ind_;
  MatrixValuesType val_;
  local_ordinal_type blockSize_;
  InVecType X_;
  beta_coeff_type beta_;
  OutVecType Y_;
};

template <class AlphaCoeffType, class GraphType, class MatrixValuesType,
          class InMultiVecType, class BetaCoeffType, class OutMultiVecType>
void bcrsLocalApplyNoTrans(
    const AlphaCoeffType& alpha, const GraphType& graph,
    const MatrixValuesType& val,
    const typename std::remove_const<typename GraphType::data_type>::type
        blockSize,
    const InMultiVecType& X, const BetaCoeffType& beta,
    const OutMultiVecType& Y) {
  static_assert(Kokkos::is_view<MatrixValuesType>::value,
                "MatrixValuesType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<OutMultiVecType>::value,
                "OutMultiVecType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<InMultiVecType>::value,
                "InMultiVecType must be a Kokkos::View.");
  static_assert(static_cast<int>(MatrixValuesType::rank) == 1,
                "MatrixValuesType must be a rank-1 Kokkos::View.");
  static_assert(static_cast<int>(OutMultiVecType::rank) == 2,
                "OutMultiVecType must be a rank-2 Kokkos::View.");
  static_assert(static_cast<int>(InMultiVecType::rank) == 2,
                "InMultiVecType must be a rank-2 Kokkos::View.");

  typedef
      typename MatrixValuesType::device_type::execution_space execution_space;
  typedef typename MatrixValuesType::device_type::memory_space memory_space;
  typedef typename MatrixValuesType::const_type matrix_values_type;
  typedef typename OutMultiVecType::non_const_type out_multivec_type;
  typedef typename InMultiVecType::const_type in_multivec_type;
  typedef typename Kokkos::ArithTraits<
      typename std::decay<AlphaCoeffType>::type>::val_type alpha_type;
  typedef typename Kokkos::ArithTraits<
      typename std::decay<BetaCoeffType>::type>::val_type beta_type;
  typedef typename std::remove_const<typename GraphType::data_type>::type LO;

  constexpr bool is_builtin_type_enabled =
      std::is_arithmetic<typename InMultiVecType::non_const_value_type>::value;
  constexpr bool is_host_memory_space =
      std::is_same<memory_space, Kokkos::HostSpace>::value;
  constexpr bool use_team_policy =
      (is_builtin_type_enabled && !is_host_memory_space);

  const LO numLocalMeshRows =
      graph.row_map.extent(0) == 0
          ? static_cast<LO>(0)
          : static_cast<LO>(graph.row_map.extent(0) - 1);
  const LO numVecs = Y.extent(1);
  if (numLocalMeshRows == 0 || numVecs == 0) {
    return;  // code below doesn't handle numVecs==0 correctly
  }

  // These assignments avoid instantiating the functor extra times
  // unnecessarily, e.g., for X const vs. nonconst.  We only need the
  // X const case, so only instantiate for that case.
  in_multivec_type X_in   = X;
  out_multivec_type Y_out = Y;

  // The functor only knows how to handle one vector at a time, and it
  // expects 1-D Views.  Thus, we need to know the type of each column
  // of X and Y.
  typedef decltype(Kokkos::subview(X_in, Kokkos::ALL(), 0)) in_vec_type;
  typedef decltype(Kokkos::subview(Y_out, Kokkos::ALL(), 0)) out_vec_type;
  typedef BcrsApplyNoTransFunctor<alpha_type, GraphType, matrix_values_type,
                                  in_vec_type, beta_type, out_vec_type,
                                  use_team_policy>
      functor_type;

  auto X_0 = Kokkos::subview(X_in, Kokkos::ALL(), 0);
  auto Y_0 = Kokkos::subview(Y_out, Kokkos::ALL(), 0);

  // Compute the first column of Y.
  if constexpr (use_team_policy) {
    functor_type functor(alpha, graph, val, blockSize, X_0, beta, Y_0);
    // Built-in version uses atomic add which might not be supported from sacado
    // or any user-defined types.
    typedef Kokkos::TeamPolicy<execution_space> policy_type;
    policy_type policy(1, 1);
#if defined(KOKKOS_ENABLE_CUDA)
    constexpr bool is_cuda = std::is_same<execution_space, Kokkos::Cuda>::value;
#else
    constexpr bool is_cuda = false;
#endif  // defined(KOKKOS_ENABLE_CUDA)
    if (is_cuda) {
      LO team_size = 8, vector_size = 1;
      if (blockSize <= 5)
        vector_size = 4;
      else if (blockSize <= 9)
        vector_size = 8;
      else if (blockSize <= 12)
        vector_size = 12;
      else if (blockSize <= 20)
        vector_size = 20;
      else
        vector_size = 20;
      policy = policy_type(numLocalMeshRows, team_size, vector_size);
    } else {
      policy = policy_type(numLocalMeshRows, 1, 1);
    }
    Kokkos::parallel_for(policy, functor);

    // Compute the remaining columns of Y.
    for (LO j = 1; j < numVecs; ++j) {
      auto X_j = Kokkos::subview(X_in, Kokkos::ALL(), j);
      auto Y_j = Kokkos::subview(Y_out, Kokkos::ALL(), j);
      functor.setX(X_j);
      functor.setY(Y_j);
      Kokkos::parallel_for(policy, functor);
    }
  } else {
    functor_type functor(alpha, graph, val, blockSize, X_0, beta, Y_0);
    Kokkos::RangePolicy<execution_space, LO> policy(0, numLocalMeshRows);
    Kokkos::parallel_for(policy, functor);
    for (LO j = 1; j < numVecs; ++j) {
      auto X_j = Kokkos::subview(X_in, Kokkos::ALL(), j);
      auto Y_j = Kokkos::subview(Y_out, Kokkos::ALL(), j);
      functor.setX(X_j);
      functor.setY(Y_j);
      Kokkos::parallel_for(policy, functor);
    }
  }
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_BSRMATRIX_SPMV_IMPL_TPETRA_HPP

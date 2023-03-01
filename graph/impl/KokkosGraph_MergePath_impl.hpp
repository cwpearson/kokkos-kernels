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

#ifndef _KOKKOSGRAPH_MERGEPATH_IMPL_HPP
#define _KOKKOSGRAPH_MERGEPATH_IMPL_HPP

#include "KokkosKernels_Iota.hpp"
#include "KokkosKernels_LowerBound.hpp"

/*! \file KokkosGraph_MergePath_impl.hpp
 *
 */

namespace KokkosGraph {
namespace Impl {

/* the index in a and b in diagonal_search where the diagonal crosses the merge
 * path
 */
template <typename a_index_type, typename b_index_type>
struct DiagonalSearchResult {
  a_index_type ai;
  b_index_type bi;
};

/*! \brief a view into the entries of the Merge Matrix along a diagonal

   \tparam AView: A Kokkos::View, the type of A
   \tparam BViewLike A Kokkos::View or an Iota, the type of B

   Example merge matrix M of two arrays
   A (vertical) and B (horizontal),
   as seen in Odeh, Green, Mwassi, Shmueli, Birk
   Merge Path - Parallel Merging Made Simple
   2012

   M[i,j] = 1 iff A[i] > B[j]

   operator(k) returns A[i] > B[j] at the kth entry of the diagonal

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
  Diagonals are indexed from the bottom-left.
  Shown on the figure above is the 1st and 5th diagonal

  The 0th diagonal D_0 has length 0
  The 1st diagonal D_1 has length 1
  The 5th diagonal D_5 has length 5
  The 9th diagonal D_9 has length 7

  D_1(0) = 1
  D_5(0..3) = 1
  D_5(4) = 0
*/
template <
typename AView,
typename BViewLike
> class MergeMatrixDiagonal {
 public:
  static_assert(AView::rank == 1, "MergeMatrixDiagonal AView must be rank 1");
  static_assert(BViewLike::rank == 1,
                "MergeMatrixDiagonal BViewLike must be rank 1");

  // implement bare minimum parts of the view interface
  enum { Rank = 1 };
  using non_const_value_type = bool;
  using AIndex = typename AView::size_type;
  using BIndex = typename BViewLike::size_type;
  using size_type = typename std::conditional<sizeof(AIndex) >= sizeof(BIndex),
                                    AIndex,
                                    BIndex>::type; // larger size_type of the two view-like types

  using result_type = DiagonalSearchResult<AIndex,
                                           BIndex>;

  KOKKOS_INLINE_FUNCTION
  MergeMatrixDiagonal(const AView &a, const BViewLike &b,
                      const size_type diagonal)
      : a_(a), b_(b), d_(diagonal) {}
  MergeMatrixDiagonal() = default;

  KOKKOS_INLINE_FUNCTION
  result_type result(const size_type &di) const noexcept {
    result_type res;
    if (0 == d_) {
      res.ai = 0;
      res.bi = 0;
      return res;
    } else {
      res = diag_to_a_b(di);
      res.ai += 1;
      return res;
    }
  }

  // compare a[i] > b[j] along diagonal at entry di
  KOKKOS_INLINE_FUNCTION
  bool operator()(const size_type di) const {
    result_type res = diag_to_a_b(di);
    if (res.ai >= AIndex(a_.size())) {
      return true;  // on the +a side out of matrix bounds is 1
    } else if (res.bi >= BIndex(b_.size())) {
      return false;  // on the +b side out of matrix bounds is 0
    } else {
      return a_(res.ai) > b_(res.bi);
    }
  }

  /*! \brief length of the diagonal

  */
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    if (d_ <= size_type(a_.size()) && d_ <= size_type(b_.size())) {
      return d_;
    } else if (d_ > size_type(a_.size()) && d_ > size_type(b_.size())) {
      // TODO: this returns nonsense if d_ happens to be outside the merge
      // matrix
      return a_.size() + b_.size() - d_;
    } else {
      return KOKKOSKERNELS_MACRO_MIN(a_.size(), b_.size());
    }
  }

 private:
  // translate an index along the diagonal to indices into a_ and b_
  KOKKOS_INLINE_FUNCTION
  result_type diag_to_a_b(const size_type &di) const noexcept {
    result_type res;
    res.ai = d_ < size_type(a_.size()) ? (d_ - 1) - di : a_.size() - 1 - di;
    res.bi = d_ < size_type(a_.size()) ? di : d_ + di - a_.size();
    return res;
  }

  AView a_;
  BViewLike b_;
  size_type d_;  // diagonal
};

/*! \brief Return the first index on diagonal \code diag
           in the merge matrix of \code a and \code b that is not 1

This is effectively a lower-bound search on the merge matrix diagonal
where the predicate is "equals 1"
*/
template <
typename AView, typename BViewLike
>
KOKKOS_INLINE_FUNCTION DiagonalSearchResult<
typename AView::size_type, typename BViewLike::size_type>
diagonal_search(
    const AView &a, const BViewLike &b,
    typename MergeMatrixDiagonal<AView, BViewLike>::size_type diag) {
  // unmanaged view types for a and b
  typedef Kokkos::View<typename AView::value_type *,
                       typename AView::device_type, Kokkos::MemoryUnmanaged>
      um_a_view;
  typedef Kokkos::View<typename BViewLike::value_type *,
                       typename BViewLike::device_type, Kokkos::MemoryUnmanaged>
      um_b_view;

  um_a_view ua(a.data(), a.size());

  // if BViewLike is an Iota, pass it on directly to MMD,
  // otherwise, create an unmanaged view of B
  typedef
      typename std::conditional<KokkosKernels::Impl::is_iota<BViewLike>::value,
                                BViewLike, um_b_view>::type b_type;

  using MMD = MergeMatrixDiagonal<um_a_view, b_type>;
  MMD mmd;
  if constexpr (KokkosKernels::Impl::is_iota<BViewLike>::value) {
    mmd = MMD(ua, b, diag);
  } else {
    b_type ub(b.data(), b.size());
    mmd = MMD(ua, ub, diag);
  }

  // returns index of the first element that does not satisfy pred(element,
  // value) our input view is the merge matrix entry along the diagonal, and we
  // want the first one that is not true. so our predicate just tells us if the
  // merge matrix diagonal entry is equal to true or not
  const typename MMD::size_type idx = KokkosKernels::lower_bound_thread(
      mmd, true, KokkosKernels::Equal<bool>());
  return mmd.result(idx);
}

template <
typename TeamMember,
typename AView,
typename BViewLike
>
KOKKOS_INLINE_FUNCTION DiagonalSearchResult<typename AView::size_type, typename BViewLike::size_type>
diagonal_search(
    const TeamMember &handle, const AView &a, const BViewLike &b,
    typename MergeMatrixDiagonal<AView, BViewLike>::size_type diag) {
  // unmanaged view types for a and b
  typedef Kokkos::View<typename AView::value_type *,
                       typename AView::device_type, Kokkos::MemoryUnmanaged>
      um_a_view;
  typedef Kokkos::View<typename BViewLike::value_type *,
                       typename BViewLike::device_type, Kokkos::MemoryUnmanaged>
      um_b_view;

  um_a_view ua(a.data(), a.size());

  // if BViewLike is an Iota, pass it on directly to MMD,
  // otherwise, create an unmanaged view of B
  typedef
      typename std::conditional<KokkosKernels::Impl::is_iota<BViewLike>::value,
                                BViewLike, um_b_view>::type b_type;

  using MMD = MergeMatrixDiagonal<um_a_view, b_type>;
  MMD mmd;
  if constexpr (KokkosKernels::Impl::is_iota<BViewLike>::value) {
    mmd = MMD(ua, b, diag);
  } else {
    b_type ub(b.data(), b.size());
    mmd = MMD(ua, ub, diag);
  }

  // returns index of the first element that does not satisfy pred(element,
  // value) our input view is the merge matrix entry along the diagonal, and we
  // want the first one that is not true. so our predicate just tells us if the
  // merge matrix diagonal entry is equal to true or not
  const typename MMD::size_type idx = KokkosKernels::lower_bound_team(
      handle, mmd, true, KokkosKernels::Equal<bool>());
  return mmd.result(idx);
}

#if 0
/* search for the intersection of diagonal d into a and b

   a and b are allowed to be different types so one of them can be an Iota

   returns DiagonalSearchResult where `ai` is the index in a and `bi` is the
   index in `b` where the diagonal crosses the merge path

   If the diagonal requested is too large, `ai` = a.size() and `bi` = b.size()
   is returned
*/
template <typename AView, typename BView>
KOKKOS_INLINE_FUNCTION
    DiagonalSearchResult<typename AView::size_type, typename BView::size_type>
    diagonal_search2(const AView &a, const BView &b,
                     typename AView::size_type diag) {
  using size_type = typename AView::size_type;
  using ordinal_type = typename AView::non_const_value_type;
  DiagonalSearchResult<typename AView::size_type, typename BView::size_type>
      res;

  if (diag >= a.size() + b.size()) {  // diagonal outside the grid
    res.ai = a.size();
    res.bi = b.size();
    return res;
  }

  size_type lo = 0;
  // hi - lo is the length of diagonal `diag`
  size_type hi;
  if (diag <= a.size() && diag <= b.size()) {
    hi = diag;
  } else if (diag > a.size() && diag > b.size()) {
    hi = a.size() + b.size() - diag;
  } else {
    hi = KOKKOSKERNELS_MACRO_MIN(a.size(), b.size());
  }

  // fprintf(stderr, "lo=%ld hi=%ld\n", lo, hi);
  // diagonal is indexed in the positive b and negative a direction
  while (hi > lo) {
    size_type mid = (lo + hi) / 2;
    size_type ai  = diag <= a.size() ? diag - mid - 1 : a.size() - mid - 1;
    size_type bi  = diag <= a.size() ? mid : diag - a.size() + mid;

    // printf("lo=%ld hi=%ld mid=%ld ai=%ld bi=%ld\n", lo, hi, mid, ai, bi);

    const ordinal_type av = a(ai);
    const ordinal_type bv = b(bi);
    // std::cerr << "av=" << av << " bv=" << bv << std::endl;

    if (av < bv) {
      hi = mid;
    } else if (av == bv) {  // when av and bv are equal, need to move along a,
                            // so bring down the high search index along diag
      hi = mid;
    } else {  // av > bv
      lo = mid + 1;
    }
  }

  res.ai = diag <= a.size() ? diag - hi : a.size() - hi;
  res.bi = diag <= a.size() ? hi : diag - a.size() + hi;

  // {
  //     DiagonalSearchResult<a_view_t, b_view_t> res2 =
  //     diagonal_search2(a,b,diag); if (res.ai != res2.ai || res.bi != res2.bi)
  //     {
  //         printf("ai diag=%d expected=%d,%d actual=%d,%d\n",
  //         int(diag), int(res.ai), int(res2.ai), int(res.bi), int(res2.bi));
  //     }
  // }

  return res;
}
#endif

/*
 */
template <typename View>
KOKKOS_INLINE_FUNCTION
    DiagonalSearchResult<typename View::size_type, typename View::size_type>
    diagonal_search(const View &a,
                    typename View::non_const_value_type totalWork,
                    typename View::size_type diag) {
  using value_type = typename View::non_const_value_type;
  using size_type  = typename View::size_type;

  KokkosKernels::Impl::Iota<value_type, size_type> iota(totalWork);
  return diagonal_search(a, iota, diag);
}

/*! \brief Provides context for where in the merge matrix a step is taking place
 */
struct StepperContext {
  using offset_type = size_t;

  offset_type ai;  //!< index into a
  offset_type bi;  //!< index into b
  offset_type pi;  //!< index into the path

  KOKKOS_INLINE_FUNCTION
  constexpr StepperContext(offset_type _ai, offset_type _bi, offset_type _pi)
      : ai(_ai), bi(_bi), pi(_pi) {}

  KOKKOS_INLINE_FUNCTION
  constexpr StepperContext() : StepperContext(0, 0, 0) {}
};

/*! \enum StepDirection
    \brief Whether the merge-path step was along the A or B list
*/
enum class StepDirection { a, b };

/*!
 * \brief Follow a merge path for two sorted input views and apply a stepper
 * function to each element.
 *
 * \tparam AView   Type of the first input view.
 * \tparam BView   Type of the second input view.
 * \tparam Stepper Type of the stepper function.
 * \tparam Ctxs    Types of the optional contexts passed to the stepper function.
 * 
 * \param a        First sorted input view.
 * \param b        Second sorted input view.
 * \param pathLength    Maximum merge path length to process.
 * \param stepper  Function to call for each path segment.
 * \param ctxs     Optional contexts to pass to the stepper function.
 *
 * Follow a merge path for two sorted input views a and b and applies a stepper
 * function at most pathLength elements. The stepper function should be
 * invokable with a StepDirection enum value, followed by 0, 1, 2 StepperContext
 * arguments.
 *
 * This generates the `step` StepperContext for each step of the merge path.
 * The stepper will be invoked as stepper(StepDirection, step, ctx...).
 */
template <typename AView, typename BView, typename Stepper, typename... Ctxs>
KOKKOS_INLINE_FUNCTION void merge_path_thread(const AView &a, const BView &b,
                                              const size_t pathLength,
                                              const Stepper &stepper,
                                              Ctxs... ctxs) {
  static_assert(AView::rank == 1, "follow_path requires rank-1 AView");
  static_assert(BView::rank == 1, "follow_path requires rank-1 BView");

  constexpr bool ONE_CTX =
      std::is_invocable<Stepper, StepDirection, StepperContext>::value;
  constexpr bool TWO_CTX =
      std::is_invocable<Stepper, StepDirection, StepperContext,
                        StepperContext>::value;
  constexpr bool THREE_CTX =
      std::is_invocable<Stepper, StepDirection, StepperContext, StepperContext,
                        StepperContext>::value;
  static_assert(ONE_CTX || TWO_CTX || THREE_CTX,
                "Stepper should be invokable with a StepDirection, then 1, 2, "
                "or 3 StepperContext arguments, for the step, thread, and team "
                "context respectively");

  static_assert(
      sizeof...(Ctxs) == 0 || sizeof...(Ctxs) == 1 || sizeof...(Ctxs) == 2,
      "Zero, one, or two contexts should be passed to merge_path_thread");

  StepperContext step;
  while (step.pi < pathLength && step.ai < a.size() && step.bi < b.size()) {
    if (a(step.ai) <= b(step.bi)) {  // step in A direction
      stepper(StepDirection::a, step, ctxs...);
      ++step.ai;
      ++step.pi;
    } else {  // step in B direction
      stepper(StepDirection::b, step, ctxs...);
      ++step.bi;
      ++step.pi;
    }
  }
  while (step.pi < pathLength && step.ai < a.size()) {
    stepper(StepDirection::a, step, ctxs...);
    ++step.ai;
    ++step.pi;
  }
  while (step.pi < pathLength && step.bi < b.size()) {
    stepper(StepDirection::b, step, ctxs...);
    ++step.bi;
    ++step.pi;
  }
}


/*! \brief Collaboratively follow a merge path for two sorted input views and
   apply a stepper function to each element. \tparam AView A Kokkos::View
    \tparam BViewLike A Kokkos::View or KokkosKernels::Impl::Iota
    \tparam Stepper Type of the stepper function.
    \tparam Ctxs Types of the optional contexts passed to the stepper function.
    \param a First sorted input view.
    \param b Second sorted input view.
    \param pathLength Maximum merge path length to process
    \param stepper Function to call for each path segment
    \param ctxs Optional contexts to pass to the stepper function.
    \param threadPathLength Maximum length each thread will process

    Collaboartively calls merge_path_thread on subsegments, adding a `thread`
   context to the stepper call that says where along the merge path that
   thread's processing begins
*/
template <typename TeamHandle, typename AView, typename BViewLike,
          typename Stepper>
KOKKOS_INLINE_FUNCTION void merge_path_team(const TeamHandle &handle,
                                            const AView &a, const BViewLike &b,
                                            const size_t pathLength,
                                            const Stepper &stepper,
                                            size_t threadPathLength) {
  static_assert(AView::rank == 1, "merge_path_team requires rank-1 a");
  static_assert(BViewLike::rank == 1, "merge_path_team requires rank-1 b");

  using a_uview_type =
      Kokkos::View<typename AView::data_type, typename AView::device_type,
                   Kokkos::MemoryUnmanaged>;

  // the "unmanaged" version depends on whether it's a View or Iota
  using b_uview_type =
      typename std::conditional<KokkosKernels::Impl::is_iota<BViewLike>::value,
                                BViewLike,
                                Kokkos::View<typename BViewLike::data_type,
                                             typename BViewLike::device_type,
                                             Kokkos::MemoryUnmanaged> >::type;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(handle, 0, handle.team_size()),
      [&](const size_t i) {
        // split up with a diagonal search
        // size_t threadPathLength =
        //     (pathLength + handle.team_size() - 1) / handle.team_size();
        const size_t diagonal = threadPathLength * i;
        if (diagonal >= pathLength) {
          return;
        }
        auto dsr  = Impl::diagonal_search(a, b, diagonal);
        size_t ai = dsr.ai;
        size_t bi = dsr.bi;

        // capture where in the team context the thread is working
        const StepperContext threadCtx(ai, bi, diagonal);

        // final piece pay be shorter
        threadPathLength =
            KOKKOSKERNELS_MACRO_MIN(threadPathLength, pathLength - diagonal);

        // take appropriate subviews of A and B according to diagonal search
        size_t nA = KOKKOSKERNELS_MACRO_MIN(threadPathLength, a.size() - ai);
        size_t nB = KOKKOSKERNELS_MACRO_MIN(threadPathLength, b.size() - bi);
        a_uview_type as(a, Kokkos::make_pair(ai, ai + nA));
        b_uview_type bs(b, Kokkos::make_pair(bi, bi + nB));

        // each thread contributes a path segment in parallel
        KokkosGraph::Impl::merge_path_thread(as, bs, threadPathLength, stepper, threadCtx);
      });
}

template <typename AView, typename BViewLike, typename Stepper>
struct FlatMergePath {
  constexpr FlatMergePath(size_t chunkSize, const AView &a,
                       const BViewLike &b, const Stepper &stepper)
      : chunkSize_(chunkSize), a_(a), b_(b), stepper_(stepper) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      const size_t i /*ith diagonal*/) const {
    const size_t diagonal = i * chunkSize_;
    if (diagonal >= a_.size() + b_.size()) {
      return;
    }

    // merge chunkSize_ values from the discovered starting place in a and b
    // where to start comparison in a and b
    auto dsr  = Impl::diagonal_search(a_, b_, diagonal);
    size_t ai = dsr.ai;
    size_t bi = dsr.bi;

    // capture where in the team context the thread is working
    const StepperContext threadCtx(ai, bi, diagonal);

    // final piece pay be shorter
    size_t chunkSize =
        KOKKOSKERNELS_MACRO_MIN(chunkSize_, a_.size() + b_.size() - diagonal);

    // may may not be able to read from a or b due to their sizes
    size_t nA = KOKKOSKERNELS_MACRO_MIN(chunkSize, a_.size() - ai);
    size_t nB = KOKKOSKERNELS_MACRO_MIN(chunkSize, b_.size() - bi);

    AView sa(a_, Kokkos::pair{ai, ai + nA});
    BViewLike sb(b_, Kokkos::pair{bi, bi + nB});
    KokkosGraph::Impl::merge_path_thread(sa, sb, chunkSize, stepper_, threadCtx);
  }

  template <typename ExecSpace>
  static void launch(const ExecSpace &space, const AView &a,
                     const BViewLike &b, const Stepper &stepper) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace,
                                   typename AView::memory_space>::accessible,
        "KokkoGraph::merge_into ExecSpace must be able to access A");
    if constexpr(!KokkosKernels::Impl::is_iota<BViewLike>()) {
      static_assert(
          Kokkos::SpaceAccessibility<ExecSpace,
                                    typename BViewLike::memory_space>::accessible,
          "KokkoGraph::merge_into ExecSpace must be able to access B");
    }

    // nothing to do
    if (0 == a.size() && 0 == b.size()) {
      return;
    }

    // one diagonal for each unit of concurrency
    const size_t chunkSize =
        (a.size() + b.size() + space.concurrency() - 1) / space.concurrency();
    const size_t numChunks = (a.size() + b.size() + chunkSize - 1) / chunkSize;

    // Do the parallel merge
    Kokkos::RangePolicy<ExecSpace> policy(space, 0, numChunks);
    Impl::FlatMergePath<AView, BViewLike, Stepper> merger(chunkSize, a, b, stepper);
    Kokkos::parallel_for("KokkosGraph::merge_into", policy, merger);
  }

  size_t chunkSize_;
  AView a_;
  BViewLike b_;
  Stepper stepper_;
};

/*! \brief Hierarchical execution on a merge path

*/
template <typename ExecSpace, typename AView, typename BViewLike, typename Stepper>
struct HierarchicalMergePath {

  using a_size_type = typename AView::size_type;
  using b_size_type = typename BViewLike::size_type;

  constexpr HierarchicalMergePath(const AView &a, const BViewLike &b,
                                  const Stepper &stepper,
                                  const size_t teamPathLength,
                                  const size_t threadPathLength)
      : a_(a),
        b_(b),
        stepper_(stepper),
        teamPathLength_(teamPathLength),
        threadPathLength_(threadPathLength) {}

  using execution_space    = ExecSpace;
  using scratch_space_type = typename execution_space::scratch_memory_space;

  using a_value_type = typename AView::non_const_value_type;
  using b_value_type = typename BViewLike::non_const_value_type;

  using team_policy_type = Kokkos::TeamPolicy<execution_space>;
  using team_member_type = typename team_policy_type::member_type;


  KOKKOS_INLINE_FUNCTION void operator()(const team_member_type &handle) const {
    using a_scratch_view =
        Kokkos::View<typename AView::non_const_data_type, scratch_space_type>;


    // determine team's position in the merge path
    size_t teamDiagonal = handle.league_rank() * teamPathLength_;
    if (teamDiagonal >= a_.size() + b_.size()) {
      return;
    }
    auto dsr = Impl::diagonal_search(a_, b_, teamDiagonal);

    // The entire path length may not be available since the
    // view dimensions may not divide evenly into teamPathLength_
    size_t nC =
        KOKKOSKERNELS_MACRO_MIN(teamPathLength_, a_.size() + b_.size() - teamDiagonal);
    a_size_type nA = KOKKOSKERNELS_MACRO_MIN(teamPathLength_, a_.size() - dsr.ai);
    b_size_type nB = KOKKOSKERNELS_MACRO_MIN(teamPathLength_, b_.size() - dsr.bi);


    if constexpr(KokkosKernels::Impl::is_iota<BViewLike>::value) {
      a_scratch_view as(handle.team_scratch(0), nA);
      BViewLike bs(dsr.bi, nB);

      // fill scratch
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(handle, 0, as.size()),
          [&](const size_t i) {
            as(i) = a_(dsr.ai + i);
          });
      handle.team_barrier();
      merge_path_team(handle, as, bs, nC, stepper_, threadPathLength_);
    } else {
      using b_scratch_view =
          Kokkos::View<typename BViewLike::non_const_data_type, scratch_space_type>;

      a_scratch_view as(handle.team_scratch(0), nA);
      b_scratch_view bs(handle.team_scratch(0), nB);

      // fill scratch
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(handle, 0,
                                  KOKKOSKERNELS_MACRO_MAX(as.size(), bs.size())),
          [&](const size_t i) {
            if (i < as.size()) {
              as(i) = a_(dsr.ai + i);
            }
            if (i < bs.size()) {
              bs(i) = b_(dsr.bi + i);
            }
          });
      handle.team_barrier();
      merge_path_team(handle, as, bs, nC, stepper_, threadPathLength_);
    }
  }

  size_t team_shmem_size(int /*teamSize*/) const {
    // AView is always a Kokkos view
    size_t sz = sizeof(a_value_type) * teamPathLength_;

    // BView needs some shared memory space if it is a Kokkos View
    if constexpr(!KokkosKernels::Impl::is_iota<BViewLike>()) {
      sz += sizeof(b_value_type) * teamPathLength_;
    }
    return sz;
  }

  static void launch(const ExecSpace &space, const AView &a,
                     const BViewLike &b, const Stepper &stepper) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace,
                                   typename AView::memory_space>::accessible,
        "KokkoGraph::merge_into ExecSpace must be able to access A");

    if constexpr(!KokkosKernels::Impl::is_iota<BViewLike>()) {
      static_assert(
          Kokkos::SpaceAccessibility<ExecSpace,
                                    typename BViewLike::memory_space>::accessible,
          "KokkoGraph::merge_into ExecSpace must be able to access B");
    }

    // nothing to do
    if (0 == a.size() && 0 == b.size()) {
      return;
    }

    // choose the team path length based on resources
    const size_t teamPathLength = 256;
    const size_t leagueSize = (a.size() + b.size() + teamPathLength - 1) / teamPathLength;
    const size_t teamSize   = 128;
    const size_t threadPathLength = (teamPathLength + teamSize - 1) / teamSize;

    // Do the parallel merge
    team_policy_type policy(space, leagueSize, teamSize);
    HierarchicalMergePath mergePath(a, b, stepper, teamPathLength, threadPathLength);
    Kokkos::parallel_for("KokkosGraph::HierarchicalMergePath", policy, mergePath);
  }

  AView a_;
  BViewLike b_;
  Stepper stepper_;
  size_t teamPathLength_;
  size_t threadPathLength_;
};




}  // namespace Impl
}  // namespace KokkosGraph

#endif  // _KOKKOSGRAPH_MERGEPATH_IMPL_HPP
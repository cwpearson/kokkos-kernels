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

#ifndef _KOKKOSGRAPH_MERGEPATH_HPP
#define _KOKKOSGRAPH_MERGEPATH_HPP

#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "KokkosKernels_SimpleUtils.hpp"

#include "KokkosGraph_MergePath_impl.hpp"

/*! \file KokkosGraph_MergePath.hpp
    \brief Provides a MergePath abstraction and related operations

    The "merge matrix" M of two sorted lists A and B has M[i,j] = 1 iff A[i] >
   B[j], and 0 otherwise.

    \verbatim
       0 1 2 3 B
      ________
    2| 1 1 0 0
    2| 1 1 0 0
    2| 1 1 0 0
    2| 1 1 0 0
    4| 1 1 1 1
    A
    \endverbatim

    The merge path follows the boundaries between the 0s and 1s.

    \verbatim
       0 1 2 3 B
      ________
      *->->
    2| 1 1|0 0
          v
    2| 1 1|0 0
          v
    2| 1 1|0 0
          v
    2| 1 1|0 0
          v->->
    4| 1 1 1 1|
    A         v
    \endverbatim

    This is equivalent to answering the question, "if I have two sorted lists A
   and B, what order should I take elements from them to produce a merged sorted
   list C", where ties are arbitrarily broken by choosing from list A.

    The length of the merge path is len(A) + len(B)

    This file provides functions that abstract over the merge path, calling a \c
   Stepper at each step along the merge path. The stepper is provided with a
   direction, as well as an index into list A, list B, and the number of steps
   along the merge path

    For the merge path shown above, the following Stepper calls are made:

    \verbatim
    stepper(b, {0,0,0})
    stepper(b, {0,1,1}) // step in b direction, +1 b, +1 path index
    stepper(a, {1,1,2}) // step in a direction, +1 a, +1 path index
    stepper(a, {2,1,3})
    stepper(a, {3,1,4})
    stepper(a, {4,1,5})
    stepper(b, {4,2,6})
    stepper(b, {4,3,7})
    stepper(a, {5,3,8}) // path length is 9, 9 steps have been made
    \endverbatim

    In practice, the indices into A, B, and the merge path are provided through
   \c StepperContext structs, and the direction through a \c StepDirection enum.

    The merge path is traversed hierarchically: each Team traverses a chunk, and
   each thread within the team a small part of the team's chunk. Therefore, the
   \c Stepper takes two contexts, a `step` and a `thread` context. The `step`
   context tells you where within the thread's piece the current step is. The
   `thread` context tells you where within the team's piece the current thread
   is. The global location in A, B, and the path can be recovered by summing the
   corresponding \c StepperContext members.
*/

namespace KokkosGraph {

using StepperContext = Impl::StepperContext;
using StepDirection = Impl::StepDirection;




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
  return Impl::merge_path_thread(a, b, pathLength, stepper, ctxs...);
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
  return Impl::merge_path_team(handle, a, b, pathLength, stepper, threadPathLength);
}

/**
 * 
 * @brief Executes merge_path_team and computes the upper bound of the path that each thread in the team will handle.
 * 
 * @tparam TeamHandle   TeamHandle used to access the team
 * @tparam AView        Type of the first view
 * @tparam BViewLike    Type of the second view
 * @tparam Stepper      Stepper type
 * 
 * @param handle        TeamHandle used to access the team
 * @param a             First view
 * @param b             Second view
 * @param pathLength    Length of the path
 * @param stepper       Stepper type
 */
template <typename TeamHandle, typename AView, typename BViewLike,
          typename Stepper>
KOKKOS_INLINE_FUNCTION void merge_path_team(const TeamHandle &handle,
                                            const AView &a, const BViewLike &b,
                                            const size_t pathLength,
                                            const Stepper &stepper) {
  const size_t threadPathLength =
      (pathLength + handle.team_size() - 1) / handle.team_size();
  merge_path_team(handle, a, b, pathLength, stepper, threadPathLength);
}

/*!
  \brief Executes the merge-path abstraction over sorted a and b in the ExecSpace execution space. 
  \tparam ExecSpace The execution space.
  \tparam AView The view type for the sorted a array. 
  \tparam BViewLike The view type for the sorted b array.
  \tparam Stepper The functor type for the stepper function.
  \param [in] space The execution space.
  \param [in] a The sorted a array.
  \param [in] b The sorted b array.
  \param [in] stepper The stepper function.
*/
template <typename ExecSpace, typename AView, typename BViewLike, typename Stepper>
void merge_path(const ExecSpace &space, 
                const AView &a, const BViewLike &b,
                const Stepper &stepper) {
  if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<ExecSpace>()) {
    Impl::HierarchicalMergePath<ExecSpace, AView, BViewLike, Stepper>::launch(space, a, b, stepper);
  } else {
    Impl::FlatMergePath<AView, BViewLike, Stepper>::template launch<ExecSpace>(space, a, b, stepper);
  }
}

/*!
 *  \brief Executes merge_path in the execution space of AView.
 *
 *  \param [in] a AView object.
 *  \param [in] b BViewLike object.
 *  \param [in] stepper Stepper object.
 *
 *  This function will execute merge_path in the execution space of AView.
 */
template <typename AView, typename BViewLike, typename Stepper>
void merge_path(const AView &a, const BViewLike &b,
                const Stepper &stepper) {
  using execution_space = typename AView::execution_space;
  merge_path(execution_space(), a, b, stepper);
}

}  // namespace KokkosGraph

#endif  // _KOKKOSGRAPH_MERGEPATH_HPP
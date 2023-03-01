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

#ifndef _KOKKOSGRAPH_LOADBALANCE_HPP
#define _KOKKOSGRAPH_LOADBALANCE_HPP

#include <Kokkos_Core.hpp>

#include "KokkosKernels_SimpleUtils.hpp"

#include "KokkosGraph_LoadBalance_impl.hpp"
#include "KokkosGraph_MergePath.hpp"

/*! \file KokkosGraph_LoadBalance.hpp
    \brief Implementation of global, team, and thread load-balance

    Load-balancing is concerned with work-item from a View of variable-sized tasks to the task that owns that work-item.
    
    The user must specify
    * Task: an integral type, counting the number of work-items in the task
    * Task View: a Kokkos::View<Task, ...>, representing variable-sized tasks
    * TaskIndex: an integral type large enough to represent the length of the Task View, used for
      indexing the Task View. May be less than size_t 

    Load-balancing will return result Views of length sum(TaskView), comprising
    * Tasks, a View<TaskIndex> saying which task the work-item came from
    * Ranks, a View<Task> giving each work-item in a task a unique number

    For example:

    Consider a Task View specifying a 0th task with size 2 (2 work items), a first task with size 1, a second with 0, and a third with 3.
    Since there are 4 tasks, we can safely use an `int` to index them
    We will allow each task to hold 2^64-1 work items, so use uint64_t for that

    \verbatim
    using Task = uint64_t;
    using TaskIndex = int;
    Kokkos::View<Task> taskSizes = {2, 1, 0, 3};
    \endverbatim
    Note that the TaskIndex does not appear in the Task View definition.
    It just reduces the data path sizes in the kernel.
    This is a total of 6 = 2 + 1 + 0 + 3 work-items.

    Load-balance will produce Tasks and Ranks views, each length 6, specifying the task and rank within the task for each of the six work items.
    The first two work-items are the 0th and 1st work-items in task 0. The next work item is
   the 0th work item in task 1, and the final three work-items are numbers 0, 1,
   and 2 in task 3.
   The Tasks view holds TaskIndex, because it refers to an entry of the input Task View.
   The Ranks view holds Tasks, because it counts work-items the same way the input Task View did.

    \verbatim
    View<TaskIndex> tasks = {0,0,1,3,3,3}
    View<Task> ranks = {0,1,0,0,1,2}
    \endverbatim

*/

namespace KokkosGraph {

/*! \struct LoadBalanceResult A load-balance function result, where work-item
   `i` came from Task = \c tasks(i) and was the `ranks(i)`-th work-item in that
   task


   The result of calling load-balance on a TaskSizesView of n tasks, where each
   entry in the view corresponds to the number of work-items (or size) of the
   task, `tasks(i)` and `ranks(i)` are the source task for work-item, and which
   work item it was within the task, respectively.

   For example, if the task-sizes view was {2, 1, 0, 3}
   tasks = {0, 0, 1, 3, 3, 3} and ranks = {0, 1, 0, 0, 1, 2}
   i.e., work-item 4 has index 1 in task 3

   \tparam TaskSizesView The input task-sizes view, which determines the struct
   view types

   TODO: add labels to views
*/
template <typename Task, typename TaskSizesView>
struct LoadBalanceResult {
  using tasks_view_type =
      typename Impl::ResultTypes<Task, TaskSizesView>::tasks_view_type;
  using ranks_view_type =
      typename Impl::ResultTypes<Task, TaskSizesView>::ranks_view_type;

  using unmanaged_tasks_view_type = 
      Kokkos::View<typename tasks_view_type::data_type,
      typename tasks_view_type::device_type,
      Kokkos::MemoryUnmanaged>;

  using unmanaged_ranks_view_type = 
      Kokkos::View<typename tasks_view_type::data_type,
      typename tasks_view_type::device_type,
      Kokkos::MemoryUnmanaged>;

  tasks_view_type tasks;  ///< the task each work item came from
  ranks_view_type ranks;  ///< the rank of the work-item within that task

  unmanaged_tasks_view_type unmanaged_tasks() const {
    return tasks;
  }

  unmanaged_ranks_view_type unmanaged_ranks() const {
    return ranks;
  }

  /*! \brief resize the members of this struct to support totalWork work-items
   */
  KOKKOS_INLINE_FUNCTION
  void resize_for_total_work(
      const typename TaskSizesView::non_const_value_type &totalWork) {
    Kokkos::resize(tasks, totalWork);
    Kokkos::resize(ranks, totalWork);
  }
};

/* `scanItems` is the inclusive prefix sum of the `workItems` view described for
 * `KokkosGraph::load_balance`
 */
template <typename Task, typename ExecSpace, typename TaskSizesView>
LoadBalanceResult<Task, TaskSizesView> load_balance_inclusive(
    const ExecSpace &space,
    const TaskSizesView &inclusiveScanTasks,
    const typename TaskSizesView::non_const_value_type totalWork) {
    LoadBalanceResult<Task, TaskSizesView> lbr;
    KokkosGraph::Impl::load_balance_inclusive(space, lbr.tasks, lbr.ranks, inclusiveScanTasks, totalWork);
    return lbr;
}
template <typename Task, typename TaskSizesView>
LoadBalanceResult<Task, TaskSizesView> load_balance_inclusive(
    const TaskSizesView &inclusiveScanTasks,
    const typename TaskSizesView::non_const_value_type totalWork) {
    using execution_space = typename TaskSizesView::execution_space;
    return load_balance_inclusive(execution_space(), inclusiveScanTasks, totalWork);
}

/*! \brief Produce a LoadBalanceResult from a view of task sizes

    \tparam ExecSpace Type of the execution space instance
    \tparam View Kokkos::View of task sizes

    \param[in] space Execution space instance to work in
    \param[in] taskSizes the number of work-items in each task
    \param[in] totalWork sum of the task sizes
 */
template <typename Task, typename ExecSpace, typename TaskSizesView>
LoadBalanceResult<Task, TaskSizesView> load_balance(const ExecSpace &space,
                                     const TaskSizesView &taskSizes) {
  static_assert(TaskSizesView::rank == 1,
                "KokkoGraph::load_balance requires rank-1 taskSizes");

  if (0 == taskSizes.size()) {
    return LoadBalanceResult<Task, TaskSizesView>();
  }

  TaskSizesView sum("task-sizes-prefix-sum", taskSizes.size());
  Kokkos::deep_copy(space, sum, taskSizes);
  KokkosKernels::Impl::kk_inclusive_parallel_prefix_sum<TaskSizesView, ExecSpace>(
      space, sum.size(), sum);

  // retrieve work items sum from view's memory space
  typename TaskSizesView::non_const_value_type numWorkItems;
  Kokkos::deep_copy(space, numWorkItems, Kokkos::subview(sum, sum.size() - 1));
  space.fence();

  LoadBalanceResult<Task, TaskSizesView> lbr;
  KokkosGraph::Impl::load_balance_inclusive<Task>(space, lbr.tasks, lbr.ranks, sum, numWorkItems);
  return lbr;
}

/*! \brief Produce a LoadBalanceResult from a view of task sizes

    \tparam View Kokkos::View of task sizes

    \param[in] taskSizes the number of work-items in each task
 */
template <typename Task, typename TaskSizesView>
LoadBalanceResult<Task, TaskSizesView> load_balance(const TaskSizesView &taskSizes) {
  using execution_space = typename TaskSizesView::device_type::execution_space;
  return load_balance<Task>(execution_space(), taskSizes);
}

template <typename Task, typename TaskSizesView>
LoadBalanceResult<Task, TaskSizesView> load_balance_thread(const TaskSizesView &taskSizes) {
  // Each entry in the view represents a work item, thus the size_type of the
  // view is how the work items are numbered The value of the entry is the
  // amount of work in that work item
  using work_item_type = typename TaskSizesView::non_const_value_type;

  // to be returned
  LoadBalanceResult<Task, TaskSizesView> lbr;

  // get total work size
  work_item_type totalWork = 0;
  for (size_t i = 0; i < taskSizes.size(); ++i) {
    totalWork += taskSizes(i);
  }

  // size result appropriately
  lbr.resize_for_total_work(totalWork);

  // do the load-balancing
  Impl::load_balance_into_thread(lbr.assignment, lbr.rank, taskSizes);
  return lbr;
}

/*! \brief team-collaborative load-balance into preallocated views

    \tparam TeamMember Type of `handle`
    \tparam TaskView type of `tasks` view
    \tparam RanksView type of `ranks` view
    \tparam TaskSizesView type of `inclusivePrefixSumTaskSizes`

    \param[in] handle the Kokkos::TeamPolicy handle
    \param[out] tasks entry i contains the source task for work-item i
    \param[out] ranks entry i contains the rank of work-item i in it's source
   task \param[in] inclusivePrefixSumTaskSizes inclusive prefix sum of task
   sizes

    inclusive_prefix_sum_team may be used for a team-collaborative inclusive
   prefix sum to provide `inclusivePrefixSumTaskSizes`
*/
template <typename TeamMember, typename TasksView, typename RanksView,
          typename TaskSizesView>
KOKKOS_INLINE_FUNCTION void load_balance_into_team(
    const TeamMember &handle, const TasksView &tasks, const RanksView &ranks,
    const typename TaskSizesView::non_const_value_type taskStart,
    const TaskSizesView &taskEnds) {
  return KokkosGraph::Impl::load_balance_into_team(handle, tasks, ranks, taskStart, taskEnds);
}

/*! \brief team-collaborative inclusive prefix sum

    \tparam TeamMember Type of `handle`
    \tparam OutView type of `tasks` view
    \tparam InView type of `ranks` view

    \param[in] handle the Kokkos::TeamPolicy handle
    \param[out] out the inclusive prefix sum
    \param[in] in the values to be summed
*/
template <typename TeamMember, typename OutView, typename InView>
KOKKOS_INLINE_FUNCTION void inclusive_prefix_sum_team(const TeamMember &handle,
                                                      const OutView &out,
                                                      const InView &in) {
  using index_type = typename OutView::size_type;
  using value_type = typename OutView::non_const_value_type;

  Kokkos::parallel_scan(
      Kokkos::TeamThreadRange(handle, 0, out.size()),
      KOKKOS_LAMBDA(const index_type i, value_type &partial, const bool final) {
        partial += in(i);
        if (final) {
          out(i) = partial;
        }
      });
}

/*! \brief team-collaborative load-balance
    \tparam TeamMember Type of `handle`
    \tparam View Type of `taskSizes`

    \param[in] handle the Kokkos::TeamPolicy handle
    \param[in] taskSizes The number of work-items in each task
*/
template <typename TeamMember, typename Task, typename TaskSizeView>
LoadBalanceResult<Task, TaskSizeView> load_balance_team(const TeamMember &handle,
                                          const TaskSizeView &taskSizes) {
  using work_item_type = typename TaskSizeView::non_const_value_type;

  LoadBalanceResult<Task, TaskSizeView> lbr;
  if (0 == taskSizes.size()) {
    return lbr;
  }
  TaskSizeView scratch("scratch", taskSizes.size());
  inclusive_prefix_sum_team(handle, scratch, taskSizes);
  const work_item_type totalWork = scratch(scratch.size() - 1);

  Kokkos::single(
      Kokkos::PerTeam(handle),
      KOKKOS_LAMBDA() { lbr.resize_for_total_work(totalWork); });
  load_balance_into_team(handle, lbr.tasks, lbr.ranks, scratch);
  return lbr;
}

}  // namespace KokkosGraph

#endif  // _KOKKOSGRAPH_LOADBALANCE_HPP

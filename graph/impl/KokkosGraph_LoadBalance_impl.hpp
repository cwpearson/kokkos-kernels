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

#ifndef _KOKKOSGRAPH_LOADBALANCE_IMPL_HPP
#define _KOKKOSGRAPH_LOADBALANCE_IMPL_HPP



/*! \file KokkosGraph_LoadBalance_impl.hpp
 *
 */

#include "KokkosKernels_Iota.hpp"
#include "KokkosKernels_SimpleUtils.hpp"
#include "KokkosKernels_ViewUtils.hpp"

#include "KokkosGraph_MergePath.hpp"

#include <optional>

namespace KokkosGraph {
namespace Impl {

template <typename Task, typename TaskSizesView>
struct ResultTypes {
  // each worker has a task that it came from
  using tasks_view_type = Kokkos::View<std::remove_cv_t<Task> *,
                                       typename TaskSizesView::device_type>;
  // each work item has a rank within a task.
  // ranks are necessarily counted the same way work items are,
  // since all work items may come from the same task
  using ranks_view_type =
      Kokkos::View<typename TaskSizesView::non_const_value_type *,
                   typename TaskSizesView::device_type>;
};

/*! \brief

    \tparam scanTasks inclusive prefix sum of task sizes, so the final entry is
   the total work
*/
template <typename TaskView,      // type of assignment view
          typename RankView,      // type of rank view
          typename ScanTasksView  // type of lists to merge
          >
struct FlatBalancer {
  FlatBalancer(size_t chunkSize,
           const TaskView &tasks,  // output
           const RankView &ranks,  // output
           const ScanTasksView &taskEnds)
      : chunkSize_(chunkSize),
        tasks_(tasks),
        ranks_(ranks),
        taskEnds_(taskEnds) {}

  using task_index_type = typename TaskView::non_const_value_type;
  // FIXME: these two should be the same
  using work_item_type = typename RankView::non_const_value_type;
  using scan_value_type = typename ScanTasksView::non_const_value_type;


  using iota_type = KokkosKernels::Impl::Iota<work_item_type>;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
    const work_item_type totalWork = taskEnds_(taskEnds_.size() - 1);
    const iota_type iota(totalWork);

    const size_t diagonal = i * chunkSize_;
    if (diagonal >= taskEnds_.size() + iota.size()) {
      return;
    }

    // diagonal search to figure out what part of the input the thread will
    // load-balance
    const auto dsr  = Impl::diagonal_search(taskEnds_, iota, diagonal);
    task_index_type si = dsr.ai;
    work_item_type ii = dsr.bi;

#if 0
        size_t nS = KOKKOSKERNELS_MACRO_MIN(chunkSize_, taskEnds_.size() - si);
        size_t nI = KOKKOSKERNELS_MACRO_MIN(chunkSize_, iota.size() - ii);
        auto ss = Kokkos::subview(taskEnds_, Kokkos::pair{si, si+nS});
        auto is = iota_type(nI, ii);
        StepperContext threadCtx(si, ii, diagonal);

        /* We will provide the inclusive scan of task sizes as A, and Iota as B
        Therefore, stepping in B means there's another work-item in the current task
        
        This work-item comes from task at the global index into A
        This work-item is rank bg - taskEnds(ag-1)
        */
        auto stepper = [&](const StepDirection &dir,
                            const StepperContext &step,
                            const StepperContext &thread,
                            const StepperContext &team) {
            if (StepDirection::b == dir) {
            // recover global index into a and b
            size_t ag = team.ai + thread.ai + step.ai;
            size_t bg = team.bi + thread.bi + step.bi;
            tasks_(bg) = ag;
            ranks_(bg) = bg - (ag > 0 ? taskEnds_(ag - 1) : 0);
            }
        };

        merge_path_thread(ss, is, chunkSize_, NoopStepper(), bstepper, threadCtx);
#else
    // don't make more than chunk-size steps along the merge-path
    for (size_t m = 0;
         m < chunkSize_ && si < task_index_type(taskEnds_.size()) && ii < work_item_type(iota.size());) {
      work_item_type sv = taskEnds_(si);
      work_item_type iv = iota(ii);

      if (iv < sv) {
        tasks_(ii) = si;
        ranks_(ii) =
            ii - (si > 0 ? taskEnds_(si - 1)
                         : 0);  // implicit prefixed 0 on inclusive scan
        ++ii;
        ++m;
      } else {
        ++si;
      }
    }
#endif
  }

  template <typename ExecSpace>
  static void launch(const ExecSpace &space, const TaskView &tasks,  // output
           const RankView &ranks,  // output
           const ScanTasksView &taskEnds, const size_t totalWork) {
    const size_t chunkSize =
        (totalWork + ExecSpace::concurrency() - 1) / ExecSpace::concurrency();
    FlatBalancer balancer(chunkSize, tasks, ranks, taskEnds);
    Kokkos::RangePolicy<ExecSpace> policy(
        space, 0, (taskEnds.size() + totalWork + chunkSize - 1) / chunkSize);
    Kokkos::parallel_for("KokkosGraph::FlatBalancer", policy, balancer);
  }

  size_t chunkSize_;
  TaskView tasks_;
  RankView ranks_;
  ScanTasksView taskEnds_;
};

template <typename TaskView, typename RankView, typename View, typename Iota>
KOKKOS_INLINE_FUNCTION void load_balance_into_thread(const TaskView &tasks,
                                                     const RankView &ranks,
                                                     const typename View::non_const_value_type &taskStart,
                                                     const View &taskEnds,
                                                     const size_t teamTaskOffset,
                                                     const size_t threadTaskOffset,
                                                     const Iota &workIota) {

  using work_item_type = typename View::non_const_value_type;
  using task_index_type = typename TaskView::non_const_value_type;
  task_index_type si = 0;
  work_item_type wi = 0;
  // printf("%s:%d %d %d %d %d\n", __FILE__, __LINE__,
  //   int(tasks.size()),
  //   int(ranks.size()),
  //   int(taskEnds.size()),
  //   int(workIota.size())
  // ); // DELETEME
  while (/*si < taskEnd && */ wi < work_item_type(workIota.size()) && wi < work_item_type(tasks.size()) && wi < work_item_type(ranks.size())) {
    auto sv = taskEnds(threadTaskOffset + si);
    auto wv = workIota(wi);

    if (wv < sv) {
      auto rank = wv - ((threadTaskOffset + si) > 0 ? taskEnds(threadTaskOffset + si - 1) : taskStart); // where does the task start
      auto task = teamTaskOffset + threadTaskOffset + si;
      // printf("%s:%d tasks(%d)=%d ranks(%d)=%d\n", __FILE__, __LINE__, int(wi), int(task), int(wi), int(rank)); // DELETEME
      tasks(wi) = task;
      ranks(wi) = rank;
      ++wi;
    } else {
      ++si;
    }
  }
}

template <typename TaskView, typename RankView, typename View>
KOKKOS_INLINE_FUNCTION void load_balance_into_thread(const TaskView &tasks,
                                                     const RankView &ranks,
                                                     const View &taskSizes) {
  using task_size_type = typename RankView::non_const_value_type;

  size_t i = 0;
  for (size_t task = 0; task < taskSizes.size(); ++task) {
    auto taskSize = taskSizes(task);

    // generate one piece of work for each size of the work item
    // each piece of work has a source work item and a rank within that work
    // item
    for (task_size_type rank = 0; rank < task_size_type(taskSize); ++rank) {
      tasks(i) = task;
      ranks(i) = rank;
      ++i;
    }
  }
}

template <typename TeamMember, typename TasksView, typename RanksView,
          typename TaskSizesView, typename Iota>
KOKKOS_INLINE_FUNCTION void load_balance_into_team(
    const TeamMember &handle, const TasksView &tasks, const RanksView &ranks,
    const typename TaskSizesView::non_const_value_type taskStart,
    const TaskSizesView &taskEnds, 
    const size_t teamTaskOffset,
    const Iota &iota, const size_t threadPathLength) {

    using task_index_type = typename TasksView::non_const_value_type;
    using work_item_type = typename RanksView::non_const_value_type;

    // determine threads's position in the merge path
    const size_t threadDiagonal = handle.team_rank() * threadPathLength;
    if (threadDiagonal >= taskEnds.size() + iota.size()) {
      return;
    }

    // find what region of the team's task sizes each thread will work on
    const auto dsr = Impl::diagonal_search(taskEnds, iota, threadDiagonal);
    // printf("%s:%d team=%d a[0]=%d a[%d]=%d b[0]=%d b[%d]=%d threadDiagonal=%d ai=%d, bi=%d\n", __FILE__, __LINE__, 
    //   int(handle.league_rank()),
    //   int(taskEnds(0)),
    //   int(taskEnds.size()-1),
    //   int(taskEnds(taskEnds.size()-1)),
    //   int(iota(0)),
    //   int(iota.size() - 1),
    //   int(iota(iota.size() - 1)),
    //   int(threadDiagonal),
    //   int(dsr.ai),
    //   int(dsr.bi)
    // );


    const work_item_type nWorkItems = KOKKOSKERNELS_MACRO_MIN(threadPathLength, iota.size() - dsr.bi);

    const Iota tIota(iota, Kokkos::make_pair(dsr.bi, dsr.bi+nWorkItems));
    const RanksView sranks(ranks, Kokkos::make_pair(dsr.bi, dsr.bi+nWorkItems));
    const TasksView stasks(tasks, Kokkos::make_pair(dsr.bi, dsr.bi+nWorkItems));

    // printf("%s:%d tid=%d a=%d+..., b=%d+%d\n", __FILE__, __LINE__, 
    //   int(handle.team_rank()),
    //   int(dsr.ai),
    //   int(dsr.bi),
    //   int(nWorkItems)
    // );

    KokkosGraph::Impl::load_balance_into_thread(stasks, sranks, taskStart, taskEnds, teamTaskOffset, dsr.ai, tIota);
}

/*! \brief team-collaborative load-balance into preallocated views

    \tparam TeamMember Type of `handle`
    \tparam TaskView type of `tasks` view
    \tparam RanksView type of `ranks` view
    \tparam TaskSizesView type of `inclusiveScanTaskSizes`

    \param[in] handle the Kokkos::TeamPolicy handle
    \param[out] tasks entry i contains the source task for work-item i
    \param[out] ranks entry i contains the rank of work-item i in it's source
   task \param[in] inclusiveScanTaskSizes inclusive prefix sum of task
   sizes

    inclusive_prefix_sum_team may be used for a team-collaborative inclusive
   prefix sum to provide `inclusiveScanTaskSizes`
*/
template <typename TeamMember, typename TasksView, typename RanksView,
          typename TaskSizesView>
KOKKOS_INLINE_FUNCTION void load_balance_into_team(
    const TeamMember &handle, const TasksView &tasks, const RanksView &ranks,
    const typename TaskSizesView::non_const_value_type taskStart,
    const TaskSizesView &taskEnds) {

  using task_size_type = typename TaskSizesView::non_const_value_type;
  using iota_type      = KokkosKernels::Impl::Iota<task_size_type>;

  const task_size_type totalWork =
    taskEnds(taskEnds.size() - 1);
  const iota_type workIota(totalWork);

  const size_t threadPathLength = (taskEnds.size() + totalWork + handle.team_size() - 1) / handle.team_size();
  KokkosGraph::Impl::load_balance_into_team(handle, tasks, ranks, taskStart, taskEnds, 0, workIota, threadPathLength);
}

// don't need these to be dependent on the HierarchicalBalancer template params
struct HierarchicalBalancerHints {
  std::optional<size_t> teamSize;
  std::optional<size_t> teamPathLength;
};

template <typename ExecSpace,
          typename TaskView,      // type of assignment view
          typename RankView,      // type of rank view
          typename ScanTasksView  // type of lists to merge
          >
struct HierarchicalBalancer {
  HierarchicalBalancer(const TaskView &tasks,  // output
           const RankView &ranks,  // output
           const ScanTasksView &taskEnds,
           size_t teamPathLength,
           size_t threadPathLength)
      : tasks_(tasks),
        ranks_(ranks),
        taskEnds_(taskEnds),
        teamPathLength_(teamPathLength),
        threadPathLength_(threadPathLength) {}

  using team_policy_type = Kokkos::TeamPolicy<ExecSpace>;
  using team_member_type = typename team_policy_type::member_type;
  using scratch_space_type = typename ExecSpace::scratch_memory_space;

  using task_index_type = typename TaskView::non_const_value_type;

  using work_item_type = typename RankView::non_const_value_type;
  using scan_value_type = typename ScanTasksView::non_const_value_type;
  static_assert(std::is_same_v<work_item_type, scan_value_type>, "");

  using iota_type = KokkosKernels::Impl::Iota<work_item_type>;
  using scan_scratch_view =
      Kokkos::View<scan_value_type*, scratch_space_type>;
  using tasks_scratch_view =
      Kokkos::View<task_index_type*, scratch_space_type>;
  using ranks_scratch_view =
      Kokkos::View<work_item_type*, scratch_space_type>;

  KOKKOS_INLINE_FUNCTION void operator()(const team_member_type &handle) const {
    // taskEnds is inclusive scan, so total work is the last entry
    const work_item_type totalWork = taskEnds_(taskEnds_.size() - 1);
    const iota_type iota(totalWork);

    // determine team's position in the merge path
    const size_t teamDiagonal = handle.league_rank() * teamPathLength_;
    if (teamDiagonal >= taskEnds_.size() + iota.size()) {
      return;
    }
    auto dsr = Impl::diagonal_search(taskEnds_, iota, teamDiagonal);


    // figure out how much of the merge matrix actually remains
    const task_index_type nTasks = KOKKOSKERNELS_MACRO_MIN(teamPathLength_, taskEnds_.size() - dsr.ai);
    const work_item_type nWorkItems = KOKKOSKERNELS_MACRO_MIN(teamPathLength_, iota.size() - dsr.bi);
    // if (handle.team_rank() == 0) {
    //   printf("%s:%d team %d nTasks=%d nWorkItems=%d\n", __FILE__, __LINE__, int(handle.league_rank()), int(nTasks), int(nWorkItems));
    // }

    const iota_type teamWorkIota(nWorkItems, work_item_type(dsr.bi));

    scan_scratch_view teamTaskEnds(handle.team_scratch(0), nTasks);
    tasks_scratch_view teamTasks(handle.team_scratch(0), nWorkItems);
    ranks_scratch_view teamRanks(handle.team_scratch(0), nWorkItems);

    // fill scratch with task endpoints
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(handle, 0, nTasks),
        [&](const task_index_type i) {
          teamTaskEnds(i) = taskEnds_(dsr.ai + i);
        }
    );
    // zero output scratch
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(handle, 0, nWorkItems),
        [&](const work_item_type i) {
          teamTasks(i) = 0;
          teamRanks(i) = 0;
        }
    );
    handle.team_barrier();


    // only B-direction steps will be added to the result
    // we don't know how many steps will come from A and B
    // without also doing an upper-bound diagonal search
    // but we know where those results will start being inserted
    // (dsr.bi)
    // so, just pass a subview starting at that point for the team to write into
    // auto tasks_s = Kokkos::subview(tasks_, Kokkos::pair{work_item_type(dsr.bi), work_item_type(dsr.bi+nB)});
    // auto ranks_s = Kokkos::subview(ranks_, Kokkos::pair{work_item_type(dsr.bi), work_item_type(dsr.bi+nB)});

    // since the team only has a piece of the taskEnds array, it can't tell what work-item
    // the first task started at
    const work_item_type teamTaskStart = (dsr.ai == 0) ? 0 : taskEnds_(dsr.ai - 1);

    KokkosGraph::Impl::load_balance_into_team(handle, teamTasks, teamRanks, teamTaskStart, teamTaskEnds, dsr.ai, teamWorkIota, threadPathLength_);
    handle.team_barrier();
    // FIXME, this relies on the output being zero already for tasks/ranks that are 0
    // write scratch output back out
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(handle, 0, nWorkItems), [&](const size_t i) {
        auto task = teamTasks(i);
        if (0 != task) {
          // printf("%s:%d tasks_(%d) = %d\n", __FILE__, __LINE__, int(dsr.bi+i), int(task));
          tasks_(dsr.bi + i) = task;
        }
        auto rank = teamRanks(i);
        if (0 != rank) {
          // printf("%s:%d ranks_(%d) = %d\n", __FILE__, __LINE__, int(dsr.bi+i), int(rank));
          ranks_(dsr.bi + i) = rank;
        }
      }
    );
  }

  size_t team_shmem_size(int /*teamSize*/) const {
    return sizeof(scan_value_type) * teamPathLength_ +
           sizeof(task_index_type) * teamPathLength_ +
           sizeof(work_item_type) * teamPathLength_;
  }

  static void launch(const ExecSpace &space, const TaskView &tasks,  // output
           const RankView &ranks,  // output
           const ScanTasksView &scanTasks, const size_t totalWork,
           const HierarchicalBalancerHints &hints = HierarchicalBalancerHints()) {
    // choose the team path length based on device resources

    size_t defaultTeamPathLength;
    size_t defaultTeamSize;
#if defined(KOKKOS_ARCH_VOLTA)
    defaultTeamPathLength = 1024;
    defaultTeamSize = 128;
#else
    defaultTeamPathLength = 1024;
    defaultTeamSize = 128;
#endif

    const size_t teamPathLength = hints.teamPathLength.value_or(defaultTeamPathLength);
    const size_t leagueSize = (scanTasks.size() + totalWork + teamPathLength - 1) / teamPathLength;
    const size_t teamSize   = hints.teamSize.value_or(defaultTeamSize);
    const size_t threadPathLength = (teamPathLength + teamSize - 1) / teamSize;

    if (teamPathLength < teamSize) {
      throw std::runtime_error("HierarchicalBalancer: team's path length must be >= team's size");
    }
    if (0 != teamPathLength % teamSize ) {
      throw std::runtime_error("HierarchicalBalancer: team's size must evenly divide team's path length");
    }

    // Do the parallel merge
    team_policy_type policy(space, leagueSize, teamSize);
    HierarchicalBalancer balancer(tasks, ranks, scanTasks, teamPathLength, threadPathLength);
    
    Kokkos::parallel_for("KokkosGraph::merge_into", policy, balancer);
  }

  TaskView tasks_;
  RankView ranks_;
  ScanTasksView taskEnds_;
  size_t teamPathLength_;
  size_t threadPathLength_;
};

/*!
 */
template <typename Task, typename ExecSpace, typename TaskEndsView>
void load_balance_inclusive_into(
    const ExecSpace &space,
    const typename ResultTypes<Task, TaskEndsView>::tasks_view_type &tasks, // output
    const typename ResultTypes<Task, TaskEndsView>::ranks_view_type &ranks, // output
    const TaskEndsView &taskEnds,
    const typename TaskEndsView::non_const_value_type totalWork,
    const HierarchicalBalancerHints &hints = HierarchicalBalancerHints()) {
  static_assert(TaskEndsView::rank == 1, "taskEnds must be rank 1");

  using tasks_view_type = typename ResultTypes<Task, TaskEndsView>::tasks_view_type;
  using ranks_view_type = typename ResultTypes<Task, TaskEndsView>::ranks_view_type;

  static_assert(
      Kokkos::SpaceAccessibility<ExecSpace,
                                 typename tasks_view_type::memory_space>::accessible,
      "ExecSpace must be able to access tasks");
  static_assert(
      Kokkos::SpaceAccessibility<ExecSpace,
                                 typename ranks_view_type::memory_space>::accessible,
      "ExecSpace must be able to access ranks");
  static_assert(
      Kokkos::SpaceAccessibility<ExecSpace,
                                 typename TaskEndsView::memory_space>::accessible,
      "ExecSpace must be able to access taskEnds");

  if (KokkosKernels::Impl::kk_is_gpu_exec_space<ExecSpace>()) {
    HierarchicalBalancer<ExecSpace, tasks_view_type, ranks_view_type, TaskEndsView>::launch(
      space, 
      KokkosKernels::Impl::make_unmanaged(tasks), 
      KokkosKernels::Impl::make_unmanaged(ranks), 
      KokkosKernels::Impl::make_unmanaged(taskEnds), totalWork, hints);
  } else {
    FlatBalancer<tasks_view_type, ranks_view_type, TaskEndsView>::template launch<ExecSpace>(space,       
    KokkosKernels::Impl::make_unmanaged(tasks), 
      KokkosKernels::Impl::make_unmanaged(ranks), 
      KokkosKernels::Impl::make_unmanaged(taskEnds), totalWork);
  }
}

/* `scanItems` is the inclusive prefix sum of the `workItems` view described for
 * `KokkosGraph::load_balance`
 */
template <typename Task, typename ExecSpace, typename TaskSizesView>
void load_balance_inclusive(
    const ExecSpace &space,
    typename ResultTypes<Task, TaskSizesView>::tasks_view_type &tasks,
    typename ResultTypes<Task, TaskSizesView>::ranks_view_type &ranks, const TaskSizesView &taskEnds,
    const typename TaskSizesView::non_const_value_type totalWork) {
  static_assert(TaskSizesView::rank == 1, "taskEnds must be rank 1");

  Kokkos::resize(Kokkos::WithoutInitializing, tasks, totalWork);
  Kokkos::resize(Kokkos::WithoutInitializing, ranks, totalWork);
  if (0 != totalWork) {
    load_balance_inclusive_into<Task>(space, tasks, ranks, taskEnds, totalWork);
  }
}

}  // namespace Impl
}  // namespace KokkosGraph

#endif  // _KOKKOSGRAPH_LOADBALANCE_IMPL_HPP

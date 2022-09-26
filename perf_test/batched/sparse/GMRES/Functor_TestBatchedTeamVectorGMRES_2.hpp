//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType, typename KrylovHandleType, bool UsePrec>
struct Functor_TestBatchedTeamVectorGMRES_2 {
  const ValuesViewType _D;
  const ValuesViewType _diag;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team, _team_size, _vector_length;
  const int _N_iteration;
  const double _tol;
  const int _ortho_strategy;
  const int _arnoldi_level;
  const int _other_level;
  KrylovHandleType _handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES_2(
      const ValuesViewType &D, const IntView &r, const IntView &c,
      const VectorViewType &X, const VectorViewType &B, const int N_team,
      const int team_size, const int vector_length, const int N_iteration,
      const double tol, const int ortho_strategy, const int arnoldi_level,
      const int other_level, KrylovHandleType &handle)
      : _D(D),
        _r(r),
        _c(c),
        _X(X),
        _B(B),
        _N_team(N_team),
        _team_size(team_size),
        _vector_length(vector_length),
        _N_iteration(N_iteration),
        _tol(tol),
        _ortho_strategy(ortho_strategy),
        _arnoldi_level(arnoldi_level),
        _other_level(other_level),
        _handle(handle) {}

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES_2(
      const ValuesViewType &D, const ValuesViewType &diag, const IntView &r,
      const IntView &c, const VectorViewType &X, const VectorViewType &B,
      const int N_team, const int team_size, const int vector_length,
      const int N_iteration, const double tol, int ortho_strategy,
      const int arnoldi_level, const int other_level, KrylovHandleType &handle)
      : _D(D),
        _diag(diag),
        _r(r),
        _c(c),
        _X(X),
        _B(B),
        _N_team(N_team),
        _team_size(team_size),
        _vector_length(vector_length),
        _N_iteration(N_iteration),
        _tol(tol),
        _ortho_strategy(ortho_strategy),
        _arnoldi_level(arnoldi_level),
        _other_level(other_level),
        _handle(handle) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix = _handle.first_index(member.league_rank());
    const int last_matrix  = _handle.last_index(member.league_rank());

    using TeamVectorCopy1D =
        KokkosBatched::TeamVectorCopy<MemberType,
                                      KokkosBatched::Trans::NoTranspose, 1>;

    auto d = Kokkos::subview(_D, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto x = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto b = Kokkos::subview(_B, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);

    using ScratchPadIntViewType =
        Kokkos::View<typename IntView::non_const_value_type *,
                     typename IntView::array_layout,
                     typename IntView::execution_space::scratch_memory_space>;
    using ScratchPadValuesViewType = Kokkos::View<
        typename ValuesViewType::non_const_value_type **,
        typename ValuesViewType::array_layout,
        typename ValuesViewType::execution_space::scratch_memory_space>;
    using Operator =
        KokkosBatched::CrsMatrix<ValuesViewType, ScratchPadIntViewType>;

    ScratchPadIntViewType tmp_1D_int(member.team_scratch(0),
                                     _r.extent(0) + _c.extent(0));

    auto r =
        Kokkos::subview(tmp_1D_int, Kokkos::make_pair(0, (int)_r.extent(0)));
    auto c = Kokkos::subview(
        tmp_1D_int,
        Kokkos::make_pair((int)_r.extent(0), (int)tmp_1D_int.extent(0)));

    TeamVectorCopy1D::invoke(member, _r, r);
    TeamVectorCopy1D::invoke(member, _c, c);
    Operator A(d, r, c);

    if (UsePrec) {
      ScratchPadValuesViewType diag(
          member.team_scratch(0), last_matrix - first_matrix, _diag.extent(1));
      using PrecOperator = KokkosBatched::JacobiPrec<ScratchPadValuesViewType>;

      KokkosBatched::TeamVectorCopy<MemberType>::invoke(
          member,
          Kokkos::subview(_diag, Kokkos::make_pair(first_matrix, last_matrix),
                          Kokkos::ALL),
          diag);
      PrecOperator P(diag);
      P.setComputedInverse();

      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<
          Operator, VectorViewType, PrecOperator, KrylovHandleType>(
          member, A, b, x, P, _handle);
    } else {
      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<
          Operator, VectorViewType>(member, A, b, x, _handle);
    }
  }

  inline double run() {
    std::string name("KokkosBatched::Test::TeamVectorGMRES");
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion(name.c_str());

    Kokkos::TeamPolicy<DeviceType> auto_policy(
        ceil(1. * _D.extent(0) / _N_team), Kokkos::AUTO(), Kokkos::AUTO());
    Kokkos::TeamPolicy<DeviceType> tuned_policy(
        ceil(1. * _D.extent(0) / _N_team), _team_size, _vector_length);
    Kokkos::TeamPolicy<DeviceType> policy;

    if (_team_size < 1)
      policy = auto_policy;
    else
      policy = tuned_policy;

    int maximum_iteration = _handle.get_max_iteration();

    _handle.set_memory_strategy(1);

    _handle.tmp_view = typename KrylovHandleType::TemporaryViewType(
        "", _X.extent(0), _X.extent(1) + maximum_iteration + 3);

    using ScalarType = typename ValuesViewType::non_const_value_type;
    using Layout     = typename ValuesViewType::array_layout;
    using EXSP       = typename ValuesViewType::execution_space;

    using ViewType2D = Kokkos::View<ScalarType **, Layout, EXSP>;

    size_t bytes_row_ptr = IntView::shmem_size(_r.extent(0));
    size_t bytes_col_idc = IntView::shmem_size(_c.extent(0));
    size_t bytes_2D      = ViewType2D::shmem_size(_N_team, _X.extent(1));

    size_t bytes_int  = bytes_row_ptr + bytes_col_idc;
    size_t bytes_diag = bytes_2D;

    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_diag + bytes_int));

    exec_space().fence();
    timer.reset();
    Kokkos::parallel_for(name.c_str(), policy, *this);
    exec_space().fence();
    double sec = timer.seconds();
    Kokkos::Profiling::popRegion();

    return sec;
  }
};
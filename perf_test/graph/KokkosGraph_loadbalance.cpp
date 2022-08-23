/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
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
*/
#include <iostream>
#include <chrono>
#include <iomanip>

#include "KokkosGraph_LoadBalance.hpp"
#include "Kokkos_Random.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_IOUtils.hpp"

struct Result {
  double us;
  size_t numRows;
};

template <typename T, typename Device>
Result bench(const std::string &path, int nWarmup, int nIters) {
  typedef KokkosSparse::CrsMatrix<T, int, Device> Matrix;

  Matrix crs =
      KokkosSparse::Impl::read_kokkos_crst_matrix<Matrix>(path.c_str());
  Result res;
  res.numRows = crs.numRows();

  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> Duration;

  for (int i = 0; i < nWarmup; ++i) {
    KokkosGraph::load_balance_exclusive(crs.graph.row_map, crs.nnz());
  }
  Kokkos::fence();
  auto start = Clock::now();
  for (int i = 0; i < nIters; ++i) {
    KokkosGraph::load_balance_exclusive(crs.graph.row_map, crs.nnz());
  }
  Kokkos::fence();
  Duration elapsed(Clock::now() - start);

  res.us = elapsed.count() * 1e6 / nIters;
  return res;
}

void usage(char **argv) {
  std::cerr << argv[0] << " matrix.mtx [... matrix.mtx]" << std::endl;
}

int main(int argc, char **argv) {
  Kokkos::initialize();

  if (argc < 2) {
    usage(argv);
    return EXIT_FAILURE;
  }

#define DO(DEVICE, TYPE)                                  \
  {                                                       \
    for (int i = 1; i < argc; ++i) {                      \
      Result res = bench<TYPE, DEVICE>(argv[i], 10, 500); \
      std::cout << argv[i];                               \
      std::cout << "," << res.numRows;                    \
      std::cout << "," << res.us;                         \
      std::cout << std::endl;                             \
    }                                                     \
  }

#ifdef KOKKOS_ENABLE_OPENMP
  std::cout << "OpenMP (float) c=" << Kokkos::OpenMP::concurrency()
            << std::endl;
  std::cout << "path,rows,us" << std::endl;
  DO(Kokkos::OpenMP, float)
#endif

#ifdef KOKKOS_ENABLE_CUDA
  std::cout << "CUDA (float) c=" << Kokkos::Cuda::concurrency() << std::endl;
  std::cout << "path,rows,us" << std::endl;
  DO(Kokkos::Cuda, float)
#endif

  Kokkos::finalize();

  return 0;
}


#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_HIP) && __HIPCC__
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include <rocsparse/rocsparse.h>
#endif

#include <benchmark/benchmark.h>

#include <Benchmark_Context.hpp>
#include <Benchmark_Utils.hpp>

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_crs_to_bsr_impl.hpp"
#include "KokkosSparse_csr_detect_block_size.hpp"

using namespace KokkosKernelsBenchmark;

template <typename Crs>
Crs &cached_read(const fs::path &path) {

  static Crs _crs;
  static fs::path _path;

  if (_path != path) {
    _crs = KokkosSparse::Impl::read_kokkos_crst_matrix<Crs>(path.c_str());
    _path = path;
    return _crs;
  }  else {
    return _crs;
  }
}


template <typename View, typename Matrix, typename Alpha, typename Beta>
void check_correctness(benchmark::State &state, const View &y_exp,
                       const View &y_act, const Matrix &crs, const Alpha &alpha,
                       const Beta &beta, const DieOnError &die,
                       const SkipOnError &skip) {
  using execution_space = typename View::execution_space;
  using scalar_type     = typename View::non_const_value_type;
  using AT              = Kokkos::ArithTraits<scalar_type>;
  using mag_type        = typename AT::mag_type;
  using ATM             = Kokkos::ArithTraits<mag_type>;

  // max value in A
  mag_type maxA = 0;
  Kokkos::parallel_reduce(
      "maxA", Kokkos::RangePolicy<execution_space>(0, crs.nnz()),
      KOKKOS_LAMBDA(const int &i, mag_type &lmax) {
        mag_type v = AT::abs(crs.values(i));
        lmax       = lmax > v ? lmax : v;
      },
      maxA);

  double eps = AT::epsilon();
  const double max_val =
      AT::abs(beta * 1.0 + crs.numCols() * alpha * maxA * 1.0);

  auto h_exp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_exp);
  auto h_act = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_act);

  size_t err = 0;
  std::vector<std::pair<size_t, size_t>> errIdx;
  for (size_t i = 0; i < h_exp.extent(0); ++i) {
    for (size_t k = 0; k < h_exp.extent(1); ++k) {
      const mag_type error = ATM::abs(h_exp(i, k) - h_act(i, k));
      if (error > eps * max_val) {
        ++err;
        errIdx.push_back({i, k});
      }
    }
  }
  if (err > 0) {
    size_t errLimit = 100;
    std::cerr << "first " << errLimit << " errors...\n";
    std::cerr << "i\tk\texp\tact" << std::endl;
    std::cerr << "-\t-\t---\t---" << std::endl;
    for (auto [i, k] : errIdx) {
      std::cerr << i << "\t" << k << "\t" << h_exp(i, k) << "\t" << h_act(i, k)
                << std::endl;
      if (0 == --errLimit) {
        break;
      }
    }
    std::cerr << __FILE__ << ":" << __LINE__ << ": ERROR: correctness failed "
              << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << ": threshold was "
              << eps * max_val << std::endl;

    if (die) {
      exit(EXIT_FAILURE);
    } else if (skip) {
      state.SkipWithError("correctness check failed");
    }
  }
}

struct SpmvTpetra {
  template <typename Alpha, typename Matrix, typename XView, typename Beta,
            typename YView>
  static void spmv(const char *mode, const Alpha &alpha, const Matrix &crs,
                   const XView &x, const Beta &beta, const YView &y) {
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "tpetra");
    return KokkosSparse::spmv(controls, mode, alpha, crs, x, beta, y);
  }

  static std::string name() { return "tpetra"; }
};

struct SpmvApp {
  template <typename Alpha, typename Matrix, typename XView, typename Beta,
            typename YView>
  static void spmv(const char *mode, const Alpha &alpha, const Matrix &crs,
                   const XView &x, const Beta &beta, const YView &y) {
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "app");
    return KokkosSparse::spmv(controls, mode, alpha, crs, x, beta, y);
  }

  static std::string name() { return "app"; }
};

struct SpmvModifiedApp {
  template <typename Alpha, typename Matrix, typename XView, typename Beta,
            typename YView>
  static void spmv(const char *mode, const Alpha &alpha, const Matrix &crs,
                   const XView &x, const Beta &beta, const YView &y) {
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "modified_app");
    return KokkosSparse::spmv(controls, mode, alpha, crs, x, beta, y);
  }

  static std::string name() { return "modified_app"; }
};

struct SpmvDefault {
  template <typename Alpha, typename Matrix, typename XView, typename Beta,
            typename YView>
  static void spmv(const char *mode, const Alpha &alpha, const Matrix &crs,
                   const XView &x, const Beta &beta, const YView &y) {
    return KokkosSparse::spmv(mode, alpha, crs, x, beta, y);
  }

  static std::string name() { return "default"; }
};

template <typename Spmv, typename Bsr>
void run(benchmark::State &state, const Bsr &bsr, const size_t k) {
  using execution_space = typename Bsr::execution_space;
  using memory_space    = typename Bsr::memory_space;
  using scalar_type     = typename Bsr::non_const_value_type;
  using ordinal_type    = typename Bsr::non_const_ordinal_type;
  using size_type       = typename Bsr::non_const_size_type;
  using view_t          = Kokkos::View<scalar_type **, memory_space>;

  state.counters["nnz"]        = bsr.nnz();
  state.counters["num_rows"]   = bsr.numRows() * bsr.blockDim();
  state.counters["block_size"] = bsr.blockDim();
  state.counters["num_vecs"]   = k;

  view_t y_init("y_init", bsr.numRows() * bsr.blockDim(), k);
  view_t y_exp("ye", bsr.numRows() * bsr.blockDim(), k);
  view_t y_act("ya", bsr.numRows() * bsr.blockDim(), k);
  view_t x("x", bsr.numCols() * bsr.blockDim(), k);

  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  fill_random(y_init, random_pool, 0.0, 1.0);

#if 0  // SIMPLE CASE, output should be sum of each row
  Kokkos::parallel_for("x", 
    Kokkos::RangePolicy<execution_space>(0, x.size()), 
    KOKKOS_LAMBDA(const int &i) {
      for (int kk = 0; kk < k; ++kk) {
       x(i, kk) = 1.0;
      }
    }
  );
  scalar_type alpha = 1;
  scalar_type beta = 0;
#else
  fill_random(x, random_pool, 0.0, 1.0);
  scalar_type alpha = 1; // App only supports alpha=1
  scalar_type beta  = 0; // App only supports beta=0
#endif

  Kokkos::deep_copy(y_act, y_init);
  Kokkos::deep_copy(y_exp, y_init);

  const char *mode = KokkosSparse::NoTranspose;

  // test the SpMV against whatever the default is
  KokkosSparse::spmv(mode, alpha, bsr, x, beta, y_exp);
  Kokkos::fence();
  Spmv::spmv(mode, alpha, bsr, x, beta, y_act);
  Kokkos::fence();

  check_correctness(state, y_exp, y_act, bsr, alpha, beta, DieOnError(false),
                    SkipOnError(true));

  Kokkos::fence();
  for (auto _ : state) {
    Spmv::spmv(mode, alpha, bsr, x, beta, y_exp);
    Kokkos::fence();
  }

  const size_t bytesPerSpmv =
      bsr.nnz() * bsr.blockDim() * bsr.blockDim() *
          sizeof(scalar_type)                    // A values
      + bsr.nnz() * sizeof(ordinal_type)         // A col indices
      + (bsr.numRows() + 1) * sizeof(size_type)  // A row-map
      + 2 * bsr.numRows() * bsr.blockDim() * k *
            sizeof(scalar_type)  // load / store y
      + bsr.numCols() * bsr.blockDim() * k * sizeof(scalar_type)  // load x
      ;

  state.SetBytesProcessed(bytesPerSpmv * state.iterations());
}

template <typename Bsr, typename Spmv>
void read_expand_run(benchmark::State &state, const fs::path &path,
                     const size_t blockSize, const size_t k) {
  using device_type  = typename Bsr::device_type;
  using scalar_type  = typename Bsr::non_const_value_type;
  using ordinal_type = typename Bsr::non_const_ordinal_type;

  // read Crs into host memory
  using Crs = KokkosSparse::CrsMatrix<scalar_type, ordinal_type, Kokkos::HostSpace>;

  const Crs crs = cached_read<Crs>(path);
  Bsr bsr;
  try {
    bsr = KokkosSparse::Impl::expand_crs_to_bsr<Bsr>(crs, blockSize);
  } catch (std::exception &e) {
    state.SkipWithError(e.what());
    return;
  }

  run<Spmv>(state, bsr, k);
}

template <typename Bsr, typename Spmv>
void read_convert_run(benchmark::State &state, const fs::path &path,
                      const size_t blockSize, const size_t k) {
  using device_type  = typename Bsr::device_type;
  using scalar_type  = typename Bsr::non_const_value_type;
  using ordinal_type = typename Bsr::non_const_ordinal_type;

  using Crs = KokkosSparse::CrsMatrix<scalar_type, ordinal_type, Kokkos::HostSpace>;

  const Crs crs = cached_read<Crs>(path);
  Bsr bsr;
  try {
    bsr = KokkosSparse::Impl::blocked_crs_to_bsr<Bsr>(crs, blockSize);
  } catch (std::exception &e) {
    state.SkipWithError(e.what());
    return;
  }

  run<Spmv>(state, bsr, k);
}

template <typename Ordinal, typename Scalar, typename Offset, typename Device,
          typename Spmv>
void register_expand_type(const fs::path &path) {
  using Bsr = KokkosSparse::Experimental::BsrMatrix<Scalar, Ordinal, Device,
                                                    void, Offset>;
  std::vector<size_t> ks = {1, 3};
  for (size_t bs : {3, 7}) {  // block sizes
    for (size_t k : ks) {     // multivector sizes
      std::string name =
          std::string("MatrixMarketExpanded") + "/" + std::string(path.stem()) +
          "/" + as_string<Scalar>() + "/" + as_string<Ordinal>() + "/" +
          as_string<Offset>() + "/" + std::to_string(bs) + "/" +
          std::to_string(k) + "/" + Spmv::name() + "/" + as_string<Device>();
      benchmark::RegisterBenchmark(name.c_str(), read_expand_run<Bsr, Spmv>,
                                   path, bs, k)
          ->UseRealTime();
    }
  }
}

template <typename Ordinal, typename Scalar, typename Offset, typename Device,
          typename Spmv>
void register_convert_type(const fs::path &path, size_t bs) {
  using Bsr = KokkosSparse::Experimental::BsrMatrix<Scalar, Ordinal, Device,
                                                    void, Offset>;
  std::vector<size_t> ks = {1, 3};

  for (size_t k : ks) {  // multivector sizes
    std::string name = std::string("MatrixMarketConvert") + "/" +
                       std::string(path.stem()) + "/" + as_string<Scalar>() +
                       "/" + as_string<Ordinal>() + "/" + as_string<Offset>() +
                       "/" + std::to_string(bs) + "/" + std::to_string(k) +
                       "/" + Spmv::name() + "/" + as_string<Device>();
    benchmark::RegisterBenchmark(name.c_str(), read_convert_run<Bsr, Spmv>,
                                 path, bs, k)
        ->UseRealTime();
  }
}

template <typename Device>
void register_converts(const fs::path &path, const size_t bs) {
    std::cerr << "benchmarks will use detected blocksize\n";
    register_convert_type<int, float, unsigned, Device, SpmvDefault>(path,
                                                                 bs);
    register_convert_type<int, float, unsigned, Device, SpmvTpetra>(path,
                                                                bs);
    register_convert_type<int, float, unsigned, Device, SpmvApp>(path,
                                                               bs);
    register_convert_type<int, float, unsigned, Device, SpmvModifiedApp>(path,
                                                               bs);
    register_convert_type<int64_t, double, size_t, Device, SpmvDefault>(path,
                                                                 bs);
    register_convert_type<int64_t, double, size_t, Device, SpmvTpetra>(path,
                                                                bs);
    register_convert_type<int64_t, double, size_t, Device, SpmvApp>(path,
                                                               bs);
    register_convert_type<int64_t, double, size_t, Device, SpmvModifiedApp>(path,
                                                               bs);
}

template <typename Device>
void register_expands(const fs::path &path) {
    register_expand_type<int, float, unsigned, Device, SpmvDefault>(path);
    register_expand_type<int, float, unsigned, Device, SpmvTpetra>(path);
    register_expand_type<int, float, unsigned, Device, SpmvApp>(path);
    register_expand_type<int, float, unsigned, Device, SpmvModifiedApp>(path);
    register_expand_type<int64_t, double, uint64_t, Device, SpmvDefault>(path);
    register_expand_type<int64_t, double, uint64_t, Device, SpmvTpetra>(path);
    register_expand_type<int64_t, double, uint64_t, Device, SpmvApp>(path);
    register_expand_type<int64_t, double, uint64_t, Device, SpmvModifiedApp>(path);
}

template <typename Device>
void register_path(const fs::path &path) {
  using ReadScalar  = double;
  using ReadOrdinal = int64_t;
  using ReadOffset  = uint64_t;
  using Bsr = KokkosSparse::Experimental::BsrMatrix<ReadScalar, ReadOrdinal,
                                                    Device, void, ReadOffset>;
  using Crs = KokkosSparse::CrsMatrix<ReadScalar, ReadOrdinal, Device, void,
                                      ReadOffset>;

  size_t detectedSize;
  try {
    std::cerr << "read " << path << "...\n";
    const Crs crs = cached_read<Crs>(path);
    std::cerr << "detect block sizes...\n";
    detectedSize = KokkosSparse::Impl::detect_block_size(crs);
    std::cerr << "detected block size = " << detectedSize << "\n";
  } catch (const std::exception &e) {
    std::cerr << "ERROR while reading: " << e.what() << "\n"
              << "skipping!\n";
    return;
  }

  /* If a block size can be detected, just use that block size without
     expanding the matrix.
     Otherwise, expand the matrix to some arbitrary block sizes to test BSR
  */
  if (detectedSize != 1) {
    std::cerr << "benchmarks will use detected size\n";
#if defined(KOKKOS_ENABLE_CUDA)
    register_converts<Kokkos::Cuda>(path, detectedSize);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    register_converts<Kokkos::HIP>(path, detectedSize);
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    register_converts<Kokkos::Serial>(path, detectedSize);
#endif
  } else {
    std::cerr << "benchmarks will expand each non-zero into a larger block\n";
#if defined(KOKKOS_ENABLE_CUDA)
    register_expands<Kokkos::Cuda>(path);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    register_expands<Kokkos::HIP>(path);
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    register_expands<Kokkos::Serial>(path);
#endif
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMicrosecond);
  KokkosKernelsBenchmark::add_benchmark_context(true);

  for (int i = 1; i < argc; ++i) {
#if defined(KOKKOS_ENABLE_CUDA)
    register_path<Kokkos::Cuda>(argv[i]);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    register_path<Kokkos::HIP>(argv[i]);
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    register_path<Kokkos::Serial>(argv[i]);
#endif
  }

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
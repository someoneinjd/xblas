#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the HERK implementation below.
#include "../reconfigurable_matmul/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for HERK. We choose the USM version of oneMKL DPC++/SYCL interface (https://oneapi-src.github.io/oneMKL/domains/blas/herk.html) with the
// restriction of standard data types (s, d, c, z) only. In this case, the matrices, alpha and beta all have the same data type.
// So we define our HERK interface as a template with a single type T.
template<typename T, typename Treal>
sycl::event herk(sycl::queue &queue,
                 oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans,
                 std::int64_t n,
                 std::int64_t k,
                 Treal alpha,
                 const T *a,
                 std::int64_t lda,
                 Treal beta,
                 T *c,
                 std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {}) {
    bool TransposeA   = (trans == oneapi::mkl::transpose::T || trans == oneapi::mkl::transpose::C);
    bool TransposeB   = !(trans == oneapi::mkl::transpose::T || trans == oneapi::mkl::transpose::C);
    bool ConjugateA   = (trans == oneapi::mkl::transpose::C);;
    bool ConjugateB   = !(trans == oneapi::mkl::transpose::C);
    bool SymmetricA   = false;
    bool HermitianA   = false;
    bool UpA          = upper_lower == oneapi::mkl::uplo::U;
    bool SymmetricB   = false;
    bool HermitianB   = false;
    bool UpB          = upper_lower == oneapi::mkl::uplo::U;
    bool SymmetricC   = false;
    bool HermitianC   = false;
    bool UpC          = false;
    bool HalfSpaceOut = true;

    // Check parameters for constraints set by oneMKL DPC++ interface
    _halide_user_assert(n >= 0 && k >= 0) << "n = " << n << ", k = " << k;
    _halide_user_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    _halide_user_assert(!TransposeA && lda >= k || TransposeA && lda >= n)
        << std::boolalpha << "Transpose = " << TransposeA << ", lda = " << lda << ", m = " << m << ", k = " << k;
    _halide_user_assert(ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert((std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto [KKK, JJJ, III, JJ, II, KK] = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % JJJ == 0) << "The current implementation requires that n must be a multiple of " << JJJ
                              << "(the output matrix is stored in " << JJJ << "-wide vectors for memory efficiency), but n = " << n;
    _halide_user_assert(k % KKK == 0) << "The current implementation requires that k must be a multiple of " << KKK
                              << "(the input matrices are loaded in " << KKK << "-wide vectors for memory efficiency), but k = " << k;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, TransposeA ? n : k, 1}, {0, TransposeA ? k : n, lda}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, n, ldc}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (n + (III * II - 1)) / (III * II)};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    sycl::event done;
    if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::ccccmatmul::cccsmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    A_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
    } else {
        done = t2sp::blas::row_major::zzzzmatmul::zzzdmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    A_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
    }
    for (int i = 0; i < (n + (III * II - 1)) / (III * II); i++) {
        for (int j = 0; j < (n + (JJJ * JJ - 1)) / (JJJ * JJ); j++) {
            for (int ii = 0; ii < II; ii++) {
                for (int jj = 0; jj < JJ; jj++) {
                    for (int iii = 0; iii < III; iii++) {
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            int total_i = iii + III * ii + III * II * i;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            if (total_i < m && total_j < n) {
                                if (j < i)
                                    c[total_j + total_i * ldc] = std::conj(Output_buffer(iii, ii, jj, jjj, i, j));
                                else
                                    c[total_j + total_i * ldc] = Output_buffer(jjj, jj, ii, iii, j, i);
                            }
                        }
                    }
                }
            }
        }
    }
    return done;
}

} // namespace t2sp::blas::row_major

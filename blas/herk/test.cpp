#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include "mkl_cblas.h"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "./api.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

sycl::device d{sycl::cpu_selector_v};
std::vector<sycl::device*> devices{&d};

namespace {

template <typename T, typename Treal>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower,
         oneapi::mkl::transpose trans, int n, int k, int lda, int ldc, Treal alpha, Treal beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HERK:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    queue fpga_queue(sycl::ext::intel::fpga_emulator_selector_v, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<T, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<T, usm::alloc::shared, 64>(cxt, *dev);
    vector<T, decltype(ua)> A(ua);
    vector<T, decltype(uc)> C(ua);
    rand_matrix(A, layout, trans, n, k, lda);
    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, n, n, ldc);

    auto C_ref = C;

    // Call MKL HERK.
    oneapi::mkl::blas::row_major::herk(main_queue, upper_lower, trans, n, k, alpha, A.data(), lda, 
                                       beta, C_ref.data(), ldc, dependencies).wait();

    // Call T2SP HERK
    try {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                throw oneapi::mkl::unimplemented{"Unkown", "Unkown"};
                break;
            case oneapi::mkl::layout::row_major:
                done = t2sp::blas::row_major::herk(fpga_queue, upper_lower, trans, n, k, alpha, A.data(), lda,
                                                   beta, C_ref.data(), ldc, dependencies);
                break;
            default: break;
        }
        done.wait();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HERK:\n" << e.what() << std::endl;
        print_error_code(e);
    }
    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HERK:\n" << error.what() << std::endl;
    }

    return 1;
}

class HerkUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(HerkUsmTests, ComplexSinglePrecision) {
    float alpha = 2.0;
    float beta = 3.0;
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP(test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP(test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP(test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::trans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP(test<std::complex<float>, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::trans, 72, 28, 101, 103, alpha, beta));
#endif
}
TEST_P(HerkUsmTests, ComplexDoublePrecision) {
    float alpha = 2.0;
    float beta = 3.0;
#ifdef T2SP_TEST_0
    EXPECT_TRUEORSKIP(test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_1)
    EXPECT_TRUEORSKIP(test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_2)
    EXPECT_TRUEORSKIP(test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::lower,
        oneapi::mkl::transpose::trans, 72, 28, 101, 103, alpha, beta));
#elif defined(T2SP_TEST_3)
    EXPECT_TRUEORSKIP(test<std::complex<double>, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::trans, 72, 28, 101, 103, alpha, beta));
#endif
}

INSTANTIATE_TEST_SUITE_P(HerkUsmTestSuite, HerkUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace

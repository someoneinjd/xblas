#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The scal API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"

using namespace std;

template <typename T, typename Ts>
void test(int N, int incx, int incy) {
    vector<T, allocator_helper<T, 64>> x, y;
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler, sycl::property::queue::enable_profiling());

    auto done = t2sp::blas::row_major::scal(q_device, N, rand_scalar<Ts>(), x.data(), incx);
    done.wait();

    // Get time in ns
    uint64_t start = done.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = done.template get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    std::cout << "Execution time in nanoseconds = " << exec_time << "\n";

    double number_ops = 0.0;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        number_ops = 2.0 * N;
    } else {
        number_ops = 8.0 * N;
    }
    std::cout << "GFLOPs: " << number_ops / exec_time << "\n";
    std::cout << "Size of vector x: " << N << "\n";
    std::cout << "Size of vector y: " << N << "\n"; 
}

int main() {
#if defined(T2SP_SSCAL)
    using test_type = float;
    using scalar_type = test_type;
#elif defined(T2SP_DSCAL)
    using test_type = double;
    using scalar_type = test_type;
#elif defined(T2SP_CSCAL)
    using test_type = std::complex<float>;
    using scalar_type = test_type;
#elif defined(T2SP_ZSCAL)
    using test_type = std::complex<double>;
    using scalar_type = test_type;
#elif defined(T2SP_CSSCAL)
    using test_type = std::complex<float>;
    using scalar_type = float;
#elif defined(T2SP_ZDSCAL)
    using test_type = std::complex<double>;
    using scalar_type = double;
#else
#error No test type (float or double or std::complex<float> or std::complex<double>) specified
#endif
    const auto KK = t2sp::blas::row_major::get_systolic_array_dimensions<test_type>();
    int64_t n = KK * 2048 * 2048;
    test<test_type, scalar_type>(n, 1, 1);
}

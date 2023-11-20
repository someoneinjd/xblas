#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "halide_runtime_etc.hpp"
#include "pipe_wrapper.hpp"
#include "complex_helper.hpp"
#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <typename... Args>
void log(Args &&...args) {
#ifndef T2SP_NDEBUG
  ((std::cout << "[INFO] ") << ... << args) << "\n";
#endif
}

using namespace sycl;
namespace t2sp::blas::row_major::svecadd {
template <typename xLoader_channel, typename yLoader_channel, typename Z_channel, int KK>
sycl::event svecadd(sycl::queue &q_device, int X_extent_0, int X_extent_1, float Alpha, float Beta) {
    std::vector<sycl::event> oneapi_kernel_events{};
    oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_Z_class>([=](){
            for (int b = 0; b < X_extent_1; b++) {
              for (int k = 0; k < (X_extent_0 + KK - 1) / KK; k++) {
                sycl::vec<float, KK> uZ_1_shreg;
                sycl::vec<float, KK> uY_shreg;
                sycl::vec<float, KK> uX_shreg;
                uX_shreg = xLoader_channel::read<>();
                uY_shreg = yLoader_channel::read<>();
                uZ_1_shreg = uX_shreg * Alpha + uY_shreg * Beta;
                Z_channel::write<>(uZ_1_shreg);
              }
            }
          }); //  h.single_task kernel_Z_class
        })); // q_device.submit
    return oneapi_kernel_events.back();
}
}

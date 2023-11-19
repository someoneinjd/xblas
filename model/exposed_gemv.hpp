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
namespace t2sp::blas::row_major::sgemv {
template <typename xFeeder_channel, typename aLoader_channel, typename Out_channel, int KK, int II, int III>
sycl::event gemv(sycl::queue &q_device, int A_extent_0, int A_extent_1) {
    using xFeeder_channel_array_t = decltype(xFeeder_channel::read<>());
    using aLoader_channel_array_t = decltype(aLoader_channel::read<>());
    using Out_channel_array_t = decltype(Out_channel::read<>());
    std::vector<sycl::event> oneapi_kernel_events{};
    oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_gemv>([=](){
            float uZ_shreg[II][III];
            float uX_shreg;
            float uZ[III];
            float uA_shreg[III];
            Out_channel_array_t Out_channel_array;
            xFeeder_channel_array_t xFeeder_channel_array;
            aLoader_channel_array_t aLoader_channel_array;
            for (int i = 0; i < A_extent_1 / (II * III); i++) {
              for (int k = 0; k < A_extent_0 / KK; k++) {
                for (int kk_ii = 0; kk_ii < (KK * II); kk_ii++) {
                  fpga_tools::UnrolledLoop<III>([&](auto iii) {
                    uZ[iii] = uZ_shreg[III - 1][iii];
                    fpga_tools::UnrolledLoop<III - 1>([&](auto l0) {
                      uZ_shreg[(III - 1) - l0][iii] = uZ_shreg[(III - 2) - l0][iii];
                    });
                    uZ_shreg[0][iii] = uZ[iii];
                  });
                  bool Out_channel_cond;
                  Out_channel_cond = 0;
                  xFeeder_channel_array = xFeeder_channel::read<>();
                  aLoader_channel_array = aLoader_channel::read<>();
                  fpga_tools::UnrolledLoop<III>([&](auto iii) {
                    uA_shreg[iii] = aLoader_channel_array.s[iii];
                    uX_shreg = iii == 0 ? xFeeder_channel_array.s : uX_shreg;
                    uX_shreg = sycl::ext::intel::fpga_reg(uX_shreg);
                    uZ_shreg[0][iii] = (kk_ii / II == 0 && k == 0 ? float_from_bits(0) : uZ_shreg[0][iii]) + uA_shreg[iii] * uX_shreg;
                    if (kk_ii / II == (KK - 1) && k == (A_extent_0 / KK) - 1) {
                      Out_channel_array.s[iii] = uZ_shreg[0][iii];
                      Out_channel_cond = 1;
                    }
                  });
                  if (Out_channel_cond) {
                    Out_channel::write<>(Out_channel_array);
                  }
              }
             }
            }
          });
        }));
    return oneapi_kernel_events.back();
}
}

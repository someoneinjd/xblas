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
namespace t2sp::blas::row_major::sdotprod {
template <typename xLoader_channel, typename yLoader_channel, typename Out_channel, int KK, int KKK>
sycl::event sdotprod(sycl::queue &q_device, int X_extent_1, bool SqrtRet, bool N) {
    std::vector<sycl::event> oneapi_kernel_events{};
    oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_Out_class>([=](){
            float Out_channel_array;
            for (int b = 0; b < X_extent_1; b++) {
              int addr_temp;
              addr_temp = 0;
              float uZ_1_shreg[KK];
              float Z_temp_shreg[KK];
              for (int k = 0; k < (X_extent_0 + (KK * KKK - 1)) / (KK * KKK); k++) {
                for (int kk = 0; kk < KK; kk++) {
                  float uZ_1;
                  uZ_1 = uZ_1_shreg[KK - 1];
                  fpga_tools::UnrolledLoop<KK - 1>([&](auto l1) {
                    uZ_1_shreg[KK - 1 - l1] = uZ_1_shreg[KK - 2 - l1];
                  });
                  uZ_1_shreg[0] = uZ_1;
                  sycl::vec<float, KKK> uY_shreg;
                  sycl::vec<float, KKK> uX_shreg;
                  uX_shreg = xLoader_channel::read<>();
                  uY_shreg = yLoader_channel::read<>();
                  float uZ_1_shreg_;
                  uZ_1_shreg_ = (k == 0 ? float_from_bits(0) : sycl::ext::intel::fpga_reg(uZ_1_shreg[0]));
                  fpga_tools::UnrolledLoop<KKK>([&](auto kkk) {
                    uZ_1_shreg_ = uZ_1_shreg_ + uX_shreg[kkk] * uY_shreg[kkk];
                    if ((kkk & ((KKK >> 1) - 1)) == (KKK >> 1) - 1)) {
                      uZ_1_shreg_ = sycl::ext::intel::fpga_reg(uZ_1_shreg_);
                    }
                  });
                  uZ_1_shreg[0] = uZ_1_shreg_;
                  fpga_tools::UnrolledLoop<KKK>([&](auto kkk) {
                    if (k == (X_extent_0 - 1) / (KK * KKK) && kkk == KKK - 1) {
                      Z_temp_shreg[addr_temp] = uZ_1_shreg[0];
                      addr_temp = addr_temp + 1;
                    }
                  });
                }
              }
              addr_temp = 0;
              float uZ_2_shreg;
              for (int kk = 0; kk < KK; kk++) {
                uZ_2_shreg = Z_temp_shreg[addr_temp] + ((kk == 0 ? float_from_bits(0) : uZ_2_shreg));
                if (kk == KK - 1) {
                  Out_channel_array = (SqrtRet ? std::sqrt(uZ_2_shreg) : uZ_2_shreg) /* conditional_sqrt_f32(SqrtRet, uZ_2_shreg) replaced */;
                }
                addr_temp = addr_temp + 1;
              }
              Out_channel_array = Out_channel_array / N;
              Out_channel::write<>(Out_channel_array);
            }
          }); //  h.single_task kernel_Out_class
        })); // q_device.submit
    return oneapi_kernel_events.back();
}
}

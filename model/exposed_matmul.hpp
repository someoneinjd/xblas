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
namespace t2sp::blas::row_major::ssssmatmul {
typedef struct {
	bool f0;
	bool f1;
	bool f2;
	bool f3;
} cgs;
template <typename SignalGenerator_channel, typename SA_channel, typename SB_channel, typename Product_channel, typename DC_channel, typename Out_channel, int III, int JJJ, int KKK>
sycl::event ssssmatmul(sycl::queue &q_device, int A_extent_1, int C_extent_0, int C_extent_1, float alpha, bool HalfSpaceOut) {
    std::vector<sycl::event> oneapi_kernel_events{};
    using SA_channel_array_t = decltype(SA_channel::read<>());
    using SB_channel_array_t = decltype(SB_channel::read<>());
    oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_Product_class>([=](){
              SB_channel_array_t SB_channel_array;
              SA_channel_array_t SA_channel_array;
              float Z_shreg[1024][JJJ][III];
              float Z_pipe_shreg[JJJ][1024 * (III - 1) + 1];
              int Z_pipe_iter;
              int Z_pipe_base;
              Z_pipe_iter = 1024 * III;
              Z_pipe_base = 0;
              while(1) {
                cgs signals = SignalGenerator_channel::read<>();
                bool signal0 = signals.f0;
                bool signal1 = signals.f1;
                bool signal2 = signals.f2;
                bool signal3 = signals.f3;
                sycl::vec<float, KKK> Y_shreg[JJJ];
                sycl::vec<float, KKK> X_shreg[III];
                float Z[JJJ][III];
                fpga_tools::UnrolledLoop<III>([&](auto iii) {
                  fpga_tools::UnrolledLoop<JJJ>([&](auto jjj) {
                    Z[jjj][iii] = Z_shreg[1023][jjj][iii];
                    #pragma unroll
                    for (int l1 = 0; l1 < 1023; l1++) {
                      Z_shreg[1023 - l1][jjj][iii] = Z_shreg[1022 - l1][jjj][iii];
                    }
                    Z_shreg[0][jjj][iii] = Z[jjj][iii];
                  });
                });
                if (signal0) {
                  SB_channel_array = SB_channel::read<>();
                  SA_channel_array = SA_channel::read<>();
                }
                fpga_tools::UnrolledLoop<III>([&](auto iii) {
                  fpga_tools::UnrolledLoop<JJJ>([&](auto jjj) {
                    X_shreg[iii] = (jjj == 0 ? SA_channel_array.template get<iii>() : X_shreg[iii]);
                    Y_shreg[jjj] = (iii == 0 ? SB_channel_array.template get<jjj>() : Y_shreg[jjj]);
                    float Z_shreg_;
                    Z_shreg_ = (signal1 ? float_from_bits(0) : sycl::ext::intel::fpga_reg(Z_shreg[0][jjj][iii]));
                    fpga_tools::UnrolledLoop<KKK>([&](auto kkk) {
                      Z_shreg_ = Z_shreg_ + X_shreg[iii][kkk] * Y_shreg[jjj][kkk];
                      if ((kkk & (KKK / 4 - 1)) == (KKK / 4 - 1)) {
                        Z_shreg_ = sycl::ext::intel::fpga_reg(Z_shreg_);
                      }
                    });
                    Z_shreg[0][jjj][iii] = Z_shreg_;
                    fpga_tools::UnrolledLoop<KKK>([&](auto kkk) {
                      if (kkk == (KKK - 1) && signal2) {
                        Z_pipe_shreg[jjj][iii * 1024] = Z_shreg[0][jjj][iii];
                      }
                    });
                  });
                });
                if (signal3) {
                  Z_pipe_base = Z_pipe_iter;
                }
                sycl::vec<float, JJJ> Product_channel_;
                fpga_tools::UnrolledLoop<JJJ>([&](auto b_62) {
                  Product_channel_[b_62] = Z_pipe_shreg[b_62][0];
                  fpga_tools::UnrolledLoop<JJJ>([&](auto b_62_dummy) {
                    Product_channel_[b_62_dummy] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Product_channel_[b_62_dummy]));
                  });
                });
                if (Z_pipe_iter < Z_pipe_base + 1024 * III) {
                  Product_channel::write<>(Product_channel_);
                }
                fpga_tools::UnrolledLoop<JJJ>([&](auto b_63) {
                  fpga_tools::UnrolledLoop<III - 1>([&](auto p_31) {
                    #pragma unroll
                    for (int l_31 = 0; l_31 < 1023; l_31++) {
                      Z_pipe_shreg[b_63][p_31 * 1024 + l_31] = Z_pipe_shreg[b_63][p_31 * 1024 + l_31 + 1];
                    }
                    Z_pipe_shreg[b_63][p_31 * 1024 + 1023] = sycl::ext::intel::fpga_reg(sycl::ext::intel::fpga_reg(Z_pipe_shreg[b_63][p_31 * 1024 + 1024]));
                  });
                });
                Z_pipe_iter = Z_pipe_iter + 1;
              }
            }); //  h.single_task kernel_Product_class
          })); // q_device.submit
          // kernel_Out
          log("kernel kernel_Out");
          oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
            h.single_task<class kernel_Out_class>([=](){
              for (int i = 0; i < (C_extent_1 + (32 * III - 1)) / (32 * III); i++) {
                for (int j = (HalfSpaceOut ? i : 0); j < (C_extent_0 + (32 * JJJ - 1)) / (32 * JJJ); j++) {
                  for (int iii = 0; iii < III; iii++) {
                    for (int ii = 0; ii < 32; ii++) {
                      for (int jj = 0; jj < 32; jj++) {
                        sycl::vec<float, JJJ> Add_shreg;
                        Add_shreg = DC_channel::read<>() + Product_channel::read<>() * alpha;
                        Out_channel::write<>(Add_shreg);
                      }
                    }
                  }
                }
              }
            }); //  h.single_task kernel_Out_class
          })); // q_device.submit
    return oneapi_kernel_events.back();
}
}

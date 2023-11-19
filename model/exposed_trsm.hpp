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
namespace t2sp::blas::row_major::strsm {
template <typename aLoader_channel, typename bLoader_channel, typename Out_channel, int I, int K, int II, int JJ>
sycl::event trsm(sycl::queue &q_device, const int B_extent_0) {
    using aLoader_channel_array_t = decltype(aLoader_channel::read<>);
    using bLoader_channel_array_t = decltype(bLoader_channel::read<>);
    using Out_channel_array_t = decltype(Out_channel::read<>);
    std::vector<sycl::event> oneapi_kernel_events{};
    oneapi_kernel_events.push_back(q_device.submit([&](sycl::handler &h){
          h.single_task<class kernel_mean>([=](){
            float uY_shreg[I + 1][II][JJ];
            float uX_shreg[2][II][JJ];
            float uA_shreg[II];
            Out_channel_array_t Out_channel_array;
            bLoader_channel_array_t bLoader_channel_array;
            aLoader_channel_array_t aLoader_channel_array;
            for (int j = 0; j < B_extent_0 / JJ; j++)  {
              for (int k = 0; k < K; k++)   {
                bLoader_channel_array = bLoader_channel::read<>();
                for (int i = 0; i < I; i++)    {
                  fpga_tools::UnrolledLoop<JJ>([&](auto jj) {
                    fpga_tools::UnrolledLoop<II>([&](auto ii) {
                      uX_shreg[1][ii][jj] = uX_shreg[0][ii][jj];
                    });
                  });
                  fpga_tools::UnrolledLoop<JJ>([&](auto jj) {
                    fpga_tools::UnrolledLoop<II>([&](auto ii) {
                      fpga_tools::UnrolledLoop<I>([&](auto l0) {
                        uY_shreg[I - l0][ii][jj] = uY_shreg[(I - 1) - l0][ii][jj];
                      });
                    });
                  });
                  aLoader_channel_array = aLoader_channel::read<>();
                  fpga_tools::UnrolledLoop<JJ>([&](auto jj) {
                    fpga_tools::UnrolledLoop<II>([&](auto ii) {
                    uA_shreg[ii] = jj == 0 ? aLoader_channel_array.s[ii] : uA_shreg[ii];
                    uA_shreg[ii] = sycl::ext::intel::fpga_reg(uA_shreg[ii]);
                    uX_shreg[0][ii][jj] = i == 0 ? (ii == 0 ? (bLoader_channel_array.s[jj] - (k == 0 ? float_from_bits(0) : uY_shreg[I][ii + 1][jj])) / uA_shreg[ii] : uX_shreg[0][ii - 1][jj]) : uX_shreg[1][ii][jj];
                    uY_shreg[0][ii][jj] = (k == 0 ? float_from_bits(0) : (ii == (II - 1) ? (i == I - 1 ? float_from_bits(0) : uY_shreg[I - 1][ii - (II - 1)][jj]) : uY_shreg[I][ii + 1][jj])) + uA_shreg[ii] * uX_shreg[0][ii][jj];
                    if (ii == II - 1 && i == I - 1)       {
                     Out_channel_array.s[jj] = uX_shreg[0][II - 1][jj];
                    }
                   });
                  });
                }
                Out_channel::write<>(Out_channel_array);
              }
            }
           });
        }));
    return oneapi_kernel_events.back();
}
}

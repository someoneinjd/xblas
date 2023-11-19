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
namespace t2sp::blas::row_major::sqrd {
template <int I, int J, int K, int batch_size = 16384, int FIXED_ITERATIONS = 64, int COLS_COMPONENT = 128>
sycl::event qrd(sycl::queue &q_device) {
    constexpr auto M_MINUS_COLS = FIXED_ITERATIONS > COLS_COMPONENT ? FIXED_ITERATIONS - COLS_COMPONENT : 0;

    constexpr auto ITERATIONS  = COLS_COMPONENT + M_MINUS_COLS + (COLS_COMPONENT + 1) * 
        COLS_COMPONENT / 2 + FIXED_ITERATIONS * (FIXED_ITERATIONS - 1) / 2 -                        
        M_MINUS_COLS * (M_MINUS_COLS - 1) / 2;

    std::vector<sycl::event> oneapi_kernel_events{};
    oneapi_kernel_events.push_back(q.submit([&](sycl::handler &h) {
            h.single_task<class kernel_qr>([=]() {
                for (int _QR_s0_batch = 0; _QR_s0_batch < batch_size;
                     _QR_s0_batch++) {
                    [[intel::fpga_memory(), intel::numbanks(32),
                      intel::bankwidth(32)]] float _temp_A[J][K];
                    [[intel::fpga_memory(), intel::numbanks(32),
                      intel::bankwidth(32)]] float _temp_Q[J][K];
                    // load matrix A
                    int _addr_load_A = _QR_s0_batch * batch_size;
                    for (int _ALoader_s0_j = 0; _ALoader_s0_j < J; _ALoader_s0_j++) {
                        for (int _ALoader_s0_k_load = 0; _ALoader_s0_k_load < K;
                             _ALoader_s0_k_load++) {
                            bool get[K];
                            #pragma unroll
                            for (int k = 0; k < K / 4; k++) {
                                get[k] = _ALoader_s0_k_load == k;
                            }
                            float tmp[4];
                            #pragma unroll
                            for (int k = 0; k < 4; k++) {
                                tmp[k] = serialized_A_device[_addr_load_A + k];
                            }
                            #pragma unroll
                            for (int k = 0; k < K / 4; k++) {
                                #pragma unroll
                                for (int t = 0; t < 4; t++) {
                                    if (get[k]) {
                                        _temp_A[_ALoader_s0_j][k * 4 + t] = tmp[t];
                                    }
                                    tmp[t] = sycl::ext::intel::fpga_reg(tmp[t]);
                                }
                            }
                            int _0 = _addr_load_A;
                            int _2 = _0 + 4;
                            _addr_load_A = _2;
                        } // for _ALoader_s0_k_load
                    }     // for _ALoader_s0_j
    
                    int _QR_s0_i = -1;
                    int _QR_s0_j = 0;
                    int _addr_store_R = _QR_s0_batch * FIXED_ITERATIONS * COLS_COMPONENT;
                    float _Xii, _i_r_Xii;
                    float _s_or_i_shreg[J];
                    float _Ai_shreg[K];
                    float _vec_ti[K];
                    [[intel::fpga_memory(), intel::numbanks(32),
                      intel::bankwidth(32)]] float _A_shreg[J][K];
                    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
                    [[intel::ivdep(FIXED_ITERATIONS)]] // NO-FORMAT: Attribute
                    for (int _QR_s0_i_j = 0; _QR_s0_i_j < ITERATIONS; _QR_s0_i_j++) {
                        float _vec_t[K];
                        float _sori_shreg[K / 4];
                        bool i_ge_0[K / 4], i_lt_0[K / 4], j_ge_i[K / 4], j_eq_i_plus_1[K / 4],
                            j_eq_i[K / 4];
                        #pragma unroll
                        for (int _QR_s0_k = 0; _QR_s0_k < K / 4; _QR_s0_k++) {
                            i_ge_0[_QR_s0_k] =
                                sycl::ext::intel::fpga_reg(_QR_s0_i >= 0);
                            i_lt_0[_QR_s0_k] =
                                sycl::ext::intel::fpga_reg(_QR_s0_i < 0);
                            j_ge_i[_QR_s0_k] =
                                sycl::ext::intel::fpga_reg(_QR_s0_j >= _QR_s0_i);
                            j_eq_i[_QR_s0_k] =
                                sycl::ext::intel::fpga_reg(_QR_s0_j == _QR_s0_i);
                            j_eq_i_plus_1[_QR_s0_k] = sycl::ext::intel::fpga_reg(
                                _QR_s0_j == _QR_s0_i + 1);
                            _sori_shreg[_QR_s0_k] =
                                sycl::ext::intel::fpga_reg(_s_or_i_shreg[_QR_s0_j]);
                        }
    
                        #pragma unroll
                        for (int _QR_s0_k = 0; _QR_s0_k < K; _QR_s0_k++) {
                            _vec_t[_QR_s0_k] = _temp_A[_QR_s0_j][_QR_s0_k];
                            if (i_ge_0[_QR_s0_k / 4]) {
                                _vec_t[_QR_s0_k] = _A_shreg[_QR_s0_j][_QR_s0_k];
                            }
                            if (j_eq_i[_QR_s0_k / 4]) {
                                _Ai_shreg[_QR_s0_k] = _vec_t[_QR_s0_k];
                            }
                        }
                        #pragma unroll
                        for (int _QR_s0_k = 0; _QR_s0_k < K; _QR_s0_k++) {
                            float _78, _79, _80;
                            if (j_eq_i[_QR_s0_k / 4]) {
                                _78 = 0.0f;
                            } else {
                                _78 = _vec_t[_QR_s0_k];
                            }
                            _79 = _Ai_shreg[_QR_s0_k];
                            if (i_lt_0[_QR_s0_k / 4]) {
                                _80 = 0.0f;
                            } else {
                                _80 = _sori_shreg[_QR_s0_k / 4];
                            }
                            float _81 = _79 * _80;
                            float _87 = _78 + _81;
                            _vec_t[_QR_s0_k] = _87;
                            if (j_ge_i[_QR_s0_k / 4]) {
                                _A_shreg[_QR_s0_j][_QR_s0_k] = _vec_t[_QR_s0_k];
                                // Output Q
                                _temp_Q[_QR_s0_j][_QR_s0_k] =
                                    _A_shreg[_QR_s0_j][_QR_s0_k];
                            }
                            if (j_eq_i_plus_1[_QR_s0_k / 4]) {
                                _vec_ti[_QR_s0_k] = _vec_t[_QR_s0_k];
                            }
                        } // for _QR_s0_k
    
                        // inner product
                        float _Xij = 0.0f;
                        #pragma unroll
                        for (int _QR_s0_k = 0; _QR_s0_k < K; _QR_s0_k++) {
                            float _113 = _vec_ti[_QR_s0_k];
                            float _114 = _vec_t[_QR_s0_k];
                            _Xij += _113 * _114;
                            //  if ((_QR_s0_k % 4) == 3)
                            //    _Xij = sycl::ext::intel::fpga_reg(_Xij);
                        } // for _QR_s0_k
                        if (_QR_s0_j == _QR_s0_i + 1) {
                            _Xii = _Xij;
                            _i_r_Xii = sycl::rsqrt(_Xij);
                        }
                        float _Sij = 0.0f - _Xij / _Xii;
                        if (_QR_s0_j == _QR_s0_i + 1) {
                            _s_or_i_shreg[_QR_s0_j] = _i_r_Xii;
                        } else {
                            _s_or_i_shreg[_QR_s0_j] = _Sij;
                        }
    
                        // Output R
                        bool _120 = (I - 1) > _QR_s0_i;
                        bool _121 = _QR_s0_i < _QR_s0_j;
                        bool _122 = _120 && _121;
                        if (_122) {
                            float _126 = _Xij;
                            float _127;
                            bool _128 = _QR_s0_j == _QR_s0_i + 1;
                            if (_128) {
                                float _129 = sycl::sqrt(_126);
                                _127 = _129;
                            } // if _128
                            else {
                                float _134 = _i_r_Xii;
                                float _135 = _126 * _134;
                                _127 = _135;
                            } // if _128 else
                            float _138 = _127;
                            serialized_R_device[_addr_store_R] = _138;
                            _addr_store_R = _addr_store_R + 1;
                        } // if _124
    
                        _QR_s0_j++;
                        if (_QR_s0_j == COLS_COMPONENT) {
                            _QR_s0_i++;
                            _QR_s0_j =
                                _QR_s0_i < (COLS_COMPONENT - FIXED_ITERATIONS)
                                    ? _QR_s0_i
                                    : (COLS_COMPONENT - FIXED_ITERATIONS);
                        }
                    }
    
                    int _addr_store_Q = _QR_s0_batch * batch_size;
                    for (int _QUnloader_s0_i = 0; _QUnloader_s0_i < I; _QUnloader_s0_i++) {
                        for (int _QUnloader_s0_k_sotre = 0;
                             _QUnloader_s0_k_sotre < K / 4; _QUnloader_s0_k_sotre++) {
                            bool get[K / 4];
                            #pragma unroll
                            for (int k = 0; k < K / 4; k++) {
                                get[k] = _QUnloader_s0_k_sotre == k;
                            }
                            float tmp[4];
                            #pragma unroll
                            for (int t = 0; t < K / 4; t++) {
                                #pragma unroll
                                for (int k = 0; k < 4; k++) {
                                    if (get[t]) {
                                        tmp[k] =
                                            _temp_Q[_QUnloader_s0_i][t * 4 + k];
                                    } else {
                                        tmp[k] = sycl::ext::intel::fpga_reg(tmp[k]);
                                    }
                                }
                            }
                            #pragma unroll
                            for (int k = 0; k < 4; k++) {
                                serialized_Q_device[_addr_store_Q + k] = tmp[k];
                            }
                            int _169 = _addr_store_Q;
                            int _170 = _169 + 4;
                            _addr_store_Q = _170;
                        } // for _QUnloader_s0_k_sotre
                    }     // for _QUnloader_s0_i
                }         // for _QR_s0_batch
            });
        }));
    return oneapi_kernel_events.back();
}

#pragma once

#include <tuple>
namespace t2sp::blas::row_major {

std::tuple<float *, int, int> cbind(sycl::queue &q_device, float *input_0,
                                    int x_0, int y_0, float *input_1, int x_1,
                                    int y_1) {
  if (x_0 != x_1)
    return nullptr;
  float *ret = sycl::malloc_device(sizeof(float) * x_0 * (y_0 + y_1), q_device);
  for (int x = 0; x < x_0; x++) {
    for (int y = 0; y < y_0; y++)
      ret[y + x * (y_0 + y_1)] = input_0[y + x * y_0];
    for (int y = 0; y < y_1; y++)
      ret[(y + y_0) + x * (y_0 + y_1)] = input_1[y + x * y_1];
  }
  sycl::free(input_0, q_device);
  sycl::free(input_1, q_device);
  return {ret, x_0, y_0 + y_1};
}

std::tuple<float *, int, int> fill(sycl::queue &q_device, float init, int x,
                                   int y) {
  float *ret = sycl::malloc_device(sizeof(float) * x * y, q_device);
  q_device.submit([&](sycl::handler &h) { h.fill(ret, init, x * y); }).wait();
  return {ret, x, y};
}

std::tuple<float *, int, int> diagMatrix(sycl::queue &q_device, float *arg, int n) {
  float *ret = sycl::malloc_device(sizeof(float) * n * n, q_device);
  q_device.memset(ret, 0, sizeof(float) * n * n, {}).wait();
  for (int i = 0; i < n; i++) ret[i + i * n] = arg[i];
  return {ret, n, n};
}

}

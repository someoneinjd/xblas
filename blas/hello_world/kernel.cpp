#include "pipe_wrapper.hpp"
#include <memory>
#include <random>

using xLoader = pipe_wrapper<class xLoader_class, sycl::float4, 256>;
using yLoader = pipe_wrapper<class yLoader_class, sycl::float4, 256>;
using zLoader = pipe_wrapper<class zLoader_class, sycl::float4, 256>;
using Out0 = pipe_wrapper<class Out0_class, sycl::float4, 256>;
struct Out_channel_array_t {
  float s;
};
using Out1 = pipe_wrapper<class Out1_class, Out_channel_array_t, 256>;

#include "axpy/exposed_funcs.hpp"
#include "dot/exposed_funcs.hpp"

float multi_kernels(sycl::queue &device, const float alpha, const float beta,
                    const float *x, const float *y, const float *z,
                    const int size) {
  std::vector<std::vector<sycl::event>> events{};

  auto device_x = sycl::malloc_device<float>(size, device);
  auto device_y = sycl::malloc_device<float>(size, device);
  auto device_z = sycl::malloc_device<float>(size, device);

  device.copy(x, device_x, size);
  device.copy(y, device_y, size);
  device.copy(z, device_z, size);
  device.wait();

  using namespace t2sp::blas::row_major;
  events.push_back(
      svecadd::xyloader<xLoader, yLoader>(device, device_x, device_y, size));
  events.push_back(
      svecadd::axpy<xLoader, yLoader, Out0>(device, alpha, beta, size));
  events.push_back(sdotprod::zloader<zLoader>(device, device_z, size, 1));
  events.push_back(sdotprod::dot<Out1, Out0, zLoader>(device, false, size, 1));
  return Out1::read<>().s;
}

int main() {
  std::vector<float> x(256), y(256), z(256);
  float alpha = 2.0f, beta = 3.0f, cpu_result = 0.0f;
  std::mt19937 gen{};
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (std::size_t i = 0; i < x.size(); i++) {
    x[i] = dis(gen);
    y[i] = dis(gen);
    z[i] = dis(gen);
    cpu_result += (alpha * x[i] + beta * y[i]) * z[i];
  }
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (exception const &e) {
        std::cout
            << "Caught asynchronous SYCL exception during multi_kernels:\n"
            << e.what() << std::endl;
      }
    }
  };

  sycl::queue fpga_queue(sycl::ext::intel::fpga_emulator_selector_v,
                         exception_handler);
  auto fpga_result =
      multi_kernels(fpga_queue, alpha, beta, x.data(), y.data(), z.data(), 256);
  if (fabs(cpu_result - fpga_result) < 0.005) {
    std::puts("Pass.");
  } else {
    std::printf("Result: %f (CPU) v.s. %f (FPGA)\n", cpu_result, fpga_result);
  }
}

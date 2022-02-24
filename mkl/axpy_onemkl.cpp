#include "common.h"

#include "cblas.h"
#include "reference_blas_templates.hpp"
#include "mkl_helper.hpp"

#include<oneapi/mkl.hpp>

#include <iostream>

// Opening cl::sycl namespace is unsupported on hipSYCL 
// (mainly due to CUDA/HIP design issues), better 
// avoid it
//using namespace cl::sycl;
namespace s = cl::sycl;

template <typename T> class VecAddKernel;

template <typename T>
class VecAddBench
{
protected:    
  //using ua_host = sycl::usm_allocator<T, sycl::usm::alloc::host, 64>;
  //using ua_device =  sycl::usm_allocator<T, sycl::usm::alloc::device, 64>;
  BenchmarkArgs args;
  sycl::context ctx;
  sycl::device dev;
  T* x_device;
  T* x_host;
  T* y_device;
  T* y_host;
  T* y_host_ref;
  int64_t* result_p_host;
  int64_t* result_p_device;


public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    ctx = args.device_queue.get_context();
    dev = args.device_queue.get_device();
    x_host = (T*)sycl::malloc_host(sizeof(T)*args.problem_size, ctx);
    x_device = (T*)sycl::malloc_device(sizeof(T)*args.problem_size, dev, ctx);
    y_host = (T*)sycl::malloc_host(sizeof(T)*args.problem_size, ctx);
    y_host_ref = (T*)sycl::malloc_host(sizeof(T)*args.problem_size, ctx);
    y_device = (T*)sycl::malloc_shared(sizeof(T)*args.problem_size, dev, ctx);
  }
  ~VecAddBench(){
    sycl::free(x_host, ctx);
    sycl::free(y_host_ref, ctx);
    sycl::free(x_device, ctx);
    sycl::free(y_host, ctx);
    sycl::free(y_device, ctx);
  }
  
  void setup() {
    // host memory intilization
    //There is a bug in oneMKL that requires the result_pointer to be shared
    //result_p_host = (int64_t*)sycl::malloc_shared(sizeof(int64_t), dev, ctx);

    for (size_t i =0; i < args.problem_size; i++) {
      x_host[i] = static_cast<T>(i);
      y_host[i] = static_cast<T>(i);
      y_host_ref[i] = static_cast<T>(i);
    }
    args.device_queue.memcpy(x_device, x_host, sizeof(T)*args.problem_size);
    args.device_queue.memcpy(y_device, y_host, sizeof(T)*args.problem_size);
    
  }

  void run(std::vector<cl::sycl::event>& events) {
    sycl::event ev = oneapi::mkl::blas::column_major::axpy(args.device_queue, args.problem_size, 1, x_device, 1, y_device, 1);
    events.push_back(ev);
  }

  bool verify(VerificationSetting &ver) {
    args.device_queue.memcpy(y_host, y_device, sizeof(T)*args.problem_size);
    float alpha = 1;
    int incx = 1;
    int size = args.problem_size;
    axpy(&size,&alpha, x_host, &incx, y_host_ref,&incx);
    args.device_queue.wait();
    return check_equal_vector(y_host, y_host_ref, size, incx, size, std::cout);

  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "VectorAddition_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  //app.run<VecAddBench<int>>();
  //app.run<VecAddBench<long long>>();  
  app.run<VecAddBench<float>>();
  //app.run<VecAddBench<double>>();
  return 0;
}

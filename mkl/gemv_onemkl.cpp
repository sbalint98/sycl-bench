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
  T* A_host, *A_device,* x_device, *x_host, *y_device, *y_host, *y_host_ref;
  int N;
  oneapi::mkl::transpose transa; 
  oneapi::mkl::layout layout; 

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    N = args.problem_size;
    ctx = args.device_queue.get_context();
    dev = args.device_queue.get_device();
    transa= oneapi::mkl::transpose::nontrans;
    layout= oneapi::mkl::layout::column_major;

    x_host = (T*)sycl::malloc_host(sizeof(T)*N, ctx);
    y_host = (T*)sycl::malloc_host(sizeof(T)*N, ctx);
    A_host = (T*)sycl::malloc_host(sizeof(T)*N*N, ctx);
    y_host_ref = (T*)sycl::malloc_host(sizeof(T)*N, ctx);

    x_device = (T*)sycl::malloc_device(sizeof(T)*N, dev, ctx);
    y_device = (T*)sycl::malloc_device(sizeof(T)*N, dev, ctx);
    A_device = (T*)sycl::malloc_device(sizeof(T)*N*N, dev, ctx);

  }
  ~VecAddBench(){
    sycl::free(y_device, ctx);
    sycl::free(y_host_ref, ctx);
    sycl::free(y_host, ctx);

    sycl::free(x_host, ctx);
    sycl::free(x_device, ctx);

    sycl::free(A_host, ctx);
    sycl::free(A_device, ctx);
  }
  
  void setup() {
    rand_matrix(A_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);
    rand_vector(x_host, N, 1);
    rand_vector(y_host, N, 1);

    args.device_queue.memcpy(y_host_ref, y_host, sizeof(T)*N);
    args.device_queue.memcpy(y_device, y_host, sizeof(T)*N);
    args.device_queue.memcpy(A_device, A_host, sizeof(T)*N*N);
    args.device_queue.memcpy(x_device, x_host, sizeof(T)*N);
    args.device_queue.wait();
  }

  void run(std::vector<cl::sycl::event>& events) {
    int m = N;
    int n = N;
    sycl::event ev = oneapi::mkl::blas::column_major::gemv(args.device_queue, transa, m, n, 1.0,
                                                             A_device, n, x_device, 1, 1.0,
                                                             y_device, 1);
    events.push_back(ev);
  }

  bool verify(VerificationSetting &ver) {
    args.device_queue.memcpy(y_host, y_device, sizeof(T)*N);
    float alpha = 1;
    int incx = 1;
    gemv(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa), &N, &N,
           &alpha, A_host, &N, x_host, &incx,
           &alpha, y_host_ref, &incx);
    args.device_queue.wait();
    return check_equal_vector(y_host, y_host_ref, N, incx, N, std::cout);

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

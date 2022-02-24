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
  T* A_host, *A_device,* B_device, *B_host, *C_device, *C_host, *C_host_ref;
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

    A_host = (T*)sycl::malloc_host(sizeof(T)*N*N, ctx);
    B_host = (T*)sycl::malloc_host(sizeof(T)*N*N, ctx);
    C_host = (T*)sycl::malloc_host(sizeof(T)*N*N, ctx);
    C_host_ref = (T*)sycl::malloc_host(sizeof(T)*N*N, ctx);

    A_device = (T*)sycl::malloc_device(sizeof(T)*N*N, dev, ctx);
    B_device = (T*)sycl::malloc_shared(sizeof(T)*N*N, dev, ctx);
    C_device = (T*)sycl::malloc_shared(sizeof(T)*N*N, dev, ctx);

  }
  ~VecAddBench(){
    sycl::free(C_device, ctx);
    sycl::free(C_host_ref, ctx);
    sycl::free(C_host, ctx);

    sycl::free(B_host, ctx);
    sycl::free(B_device, ctx);

    sycl::free(A_host, ctx);
    sycl::free(A_device, ctx);
  }
  
  void setup() {
    rand_matrix(A_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);
    rand_matrix(B_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);
    rand_matrix(C_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);


    args.device_queue.memcpy(C_host_ref, C_host, sizeof(T)*N*N);
    args.device_queue.memcpy(C_device, C_host, sizeof(T)*N*N);
    args.device_queue.memcpy(A_device, A_host, sizeof(T)*N*N);
    args.device_queue.memcpy(B_device, B_host, sizeof(T)*N*N);
    args.device_queue.wait();
    std::cout << "Finished setup" << std::endl;
  }

  void run(std::vector<cl::sycl::event>& events) {
    int m = N;
    int n = N;
    sycl::event ev = oneapi::mkl::blas::column_major::gemm(args.device_queue, transa, transa,
                                                             N, N, N,
                                                             1.0, A_device, N,
                                                             B_device, N, 1.0,
                                                             C_device, N);
    events.push_back(ev);
    std::cout << "finished run " << std::endl;
  }

  bool verify(VerificationSetting &ver) {
    args.device_queue.memcpy(C_host, C_device, sizeof(T)*N*N);
    float alpha = 1;
    int incx = 1;
    gemm(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),convert_to_cblas_trans(transa),
           &N, &N, &N,
           &alpha, A_host, &N,
           B_host, &N, &alpha,
           C_host_ref, &N);
    args.device_queue.wait();
    return check_equal_vector(C_host, C_host_ref, N*N, incx, N*N, std::cout);

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

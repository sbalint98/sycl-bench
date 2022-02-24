#include "common.h"

#include "cblas.h"
#include "reference_blas_templates.hpp"
#include "mkl_helper.hpp"
#include <rocblas.h>
#include <hip/hip_runtime.h>
#include "hip_helper.h"

#include<oneapi/mkl.hpp>

#include <iostream>

// Opening cl::sycl namespace is unsupported on hipSYCL 
// (mainly due to hip/HIP design issues), better 
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
  T* x_dev;
  T* x_host;
  T* y_dev;
  T* y_host;
  T* y_host_ref;
  int64_t* result_p_host;
  int64_t* result_p_device;
  int N;
  rocblas_handle handle;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    gpuErrchk(hipSetDevice(0));
    cublasErrchk(rocblas_create_handle(&handle));
    N = args.problem_size;
    x_host = (float*)malloc(sizeof(float)*N);
    y_host = (float*)malloc(sizeof(float)*N);
    y_host_ref = (float*)malloc(sizeof(float)*N);

    gpuErrchk(hipMalloc((void**)&y_dev, sizeof(float)*N));
    gpuErrchk(hipMalloc((void**)&x_dev, sizeof(float)*N));
  }
  ~VecAddBench(){
    free(x_host); 
    free(y_host); 
    free(y_host_ref);

     gpuErrchk(hipFree(x_dev)); 
     gpuErrchk(hipFree(y_dev)); 
  }
  
  void setup() {
    gpuErrchk(hipSetDevice(0));
    // host memory intilization
    //There is a bug in oneMKL that requires the result_pointer to be shared
    //result_p_host = (int64_t*)sycl::malloc_shared(sizeof(int64_t), dev, ctx);

    rand_vector(x_host, N, 1);
    rand_vector(y_host, N, 1);

    std::memcpy(y_host_ref, y_host, sizeof(float)*N);
    gpuErrchk(hipMemcpy((void*)y_dev, (void*)y_host, sizeof(float)*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void*)x_dev, (void*)x_host, sizeof(float)*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipDeviceSynchronize());
    
  }

  void run(std::vector<cl::sycl::event>& events) {
    float alpha = 1.0;
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    rocblas_saxpy(handle,N, &alpha,
                             x_dev, 1, y_dev, 1);
    gpuErrchk(hipDeviceSynchronize());
  }

  bool verify(VerificationSetting &ver) {
    args.device_queue.memcpy(y_host, y_dev, sizeof(T)*args.problem_size);
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

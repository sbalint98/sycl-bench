#include "common.h"

#include "cblas.h"
#include "reference_blas_templates.hpp"
#include "mkl_helper.hpp"
#include "cublas_v2.h"
#include <cuda_runtime.h>

#include<oneapi/mkl.hpp>

#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"GPUassert: %s %d\n", file, line);
      if (abort) exit(code);
   }
}


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
  T* x_dev;
  T* x_host;
  T* y_dev;
  T* y_host;
  T* y_host_ref;
  int64_t* result_p_host;
  int64_t* result_p_device;
  int N;
  cublasHandle_t handle;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    cudaSetDevice(0);
    cublasErrchk(cublasCreate(&handle));
    N = args.problem_size;
    x_host = (float*)malloc(sizeof(float)*N);
    y_host = (float*)malloc(sizeof(float)*N);
    y_host_ref = (float*)malloc(sizeof(float)*N);

    gpuErrchk(cudaMalloc((void**)&y_dev, sizeof(float)*N));
    gpuErrchk(cudaMalloc((void**)&x_dev, sizeof(float)*N));
  }
  ~VecAddBench(){
    free(x_host); 
    free(y_host); 
    free(y_host_ref);

    cudaFree(x_dev); 
    cudaFree(y_dev); 
  }
  
  void setup() {
    cudaSetDevice(0);
    // host memory intilization
    //There is a bug in oneMKL that requires the result_pointer to be shared
    //result_p_host = (int64_t*)sycl::malloc_shared(sizeof(int64_t), dev, ctx);

    rand_vector(x_host, N, 1);
    rand_vector(y_host, N, 1);

    std::memcpy(y_host_ref, y_host, sizeof(float)*N);
    gpuErrchk(cudaMemcpy((void*)y_dev, (void*)y_host, sizeof(float)*N,  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void*)x_dev, (void*)x_host, sizeof(float)*N,  cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    
  }

  void run(std::vector<cl::sycl::event>& events) {
    float alpha = 1.0;
    cublasSaxpy(handle,N, &alpha,
                             x_dev, 1, y_dev, 1);
    cudaDeviceSynchronize();
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

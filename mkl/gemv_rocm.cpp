#include "common.h"

#include "cblas.h"
#include "reference_blas_templates.hpp"
#include "mkl_helper.hpp"
#include "hip_helper.h"
#include <rocblas.h>
#include <hip/hip_runtime.h>

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
  rocblas_handle handle;
  sycl::context ctx;
  sycl::device dev;
  float *A_dev, *x_dev, *y_dev;
  float *A_host, *x_host, *y_host, *y_host_ref;
  int N;


public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    hipSetDevice(0);
    cublasErrchk(rocblas_create_handle(&handle));
    N = args.problem_size;
    A_host = (float*)malloc(sizeof(float)*N*N);
    x_host = (float*)malloc(sizeof(float)*N);
    y_host = (float*)malloc(sizeof(float)*N);
    y_host_ref = (float*)malloc(sizeof(float)*N);
    
    gpuErrchk(hipMalloc((void**)&A_dev, sizeof(float)*N*N));
    gpuErrchk(hipMalloc((void**)&y_dev, sizeof(float)*N));
    gpuErrchk(hipMalloc((void**)&x_dev, sizeof(float)*N));
  }

  ~VecAddBench(){
    free(A_host);
    free(x_host); 
    free(y_host); 
    free(y_host_ref);

    gpuErrchk(hipFree(A_dev));
    gpuErrchk(hipFree(x_dev)); 
    gpuErrchk(hipFree(y_dev)); 
  }
  
  void setup() {
    hipSetDevice(0);
   
    for(int i = 0; i < N; i++){
      x_host[i] = i;
      y_host[i] = i;
      y_host_ref[i] = i;
    }

    for(int i = 0; i < N*N; i++){
      A_host[i] = i;
    }

    gpuErrchk(hipMemcpy((void*)y_dev, (void*)y_host, sizeof(float)*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void*)x_dev, (void*)x_host, sizeof(float)*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void*)A_dev, (void*)A_host, sizeof(float)*N*N,  hipMemcpyHostToDevice));
    hipDeviceSynchronize();

  }

  void run(std::vector<cl::sycl::event>& events) {
    //hipSetDevice(0);
    float alpha = 1.0;
    cublasErrchk(rocblas_sgemv(handle, rocblas_operation_none,
                        N,N,
                        &alpha,
                        A_dev, N,
                        x_dev, 1,
                        &alpha,
                        y_dev, 1));
    hipDeviceSynchronize();
  }

  bool verify(VerificationSetting &ver) {
    gpuErrchk(hipMemcpy((void*)y_host, (void*)y_dev, sizeof(float)*N,    hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy((void*)x_host, (void*)x_dev, sizeof(float)*N,    hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy((void*)A_host, (void*)A_dev, sizeof(float)*N*N,  hipMemcpyDeviceToHost));    
    int m = args.problem_size;
    int n = args.problem_size;
    float alpha = 1;
    int incx = 1;
    int size = args.problem_size;
    ::gemv(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, &m, &n,
           &alpha, (float*)A_host,&m, x_host, &incx,
           &alpha, y_host_ref, &incx);
    //args.device_queue.wait();
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

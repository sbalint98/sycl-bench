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
  float *A_dev, *B_dev, *C_dev;
  float *A_host, *B_host, *C_host, *C_host_ref;
  int N;
  oneapi::mkl::transpose transa; 
  oneapi::mkl::layout layout; 


public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {
    gpuErrchk(hipSetDevice(0));
    cublasErrchk(rocblas_create_handle(&handle));
    N = args.problem_size;
    transa= oneapi::mkl::transpose::nontrans;
    layout= oneapi::mkl::layout::column_major;

    A_host = (float*)malloc(sizeof(float)*N*N);
    B_host = (float*)malloc(sizeof(float)*N*N);
    C_host = (float*)malloc(sizeof(float)*N*N);
    C_host_ref = (float*)malloc(sizeof(float)*N*N);
    
    gpuErrchk(hipMalloc((void**)&A_dev, sizeof(float)*N*N));
    gpuErrchk(hipMalloc((void**)&B_dev, sizeof(float)*N*N));
    gpuErrchk(hipMalloc((void**)&C_dev, sizeof(float)*N*N));
  }

  ~VecAddBench(){
    free(A_host);
    free(B_host); 
    free(C_host); 
    free(C_host_ref);

    gpuErrchk(hipFree(A_dev));
    gpuErrchk(hipFree(B_dev)); 
    gpuErrchk(hipFree(C_dev));

    cublasErrchk(rocblas_destroy_handle(handle));
  }
  
  void setup() {
    gpuErrchk(hipSetDevice(0));
   
    rand_matrix(A_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);
    rand_matrix(B_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);
    rand_matrix(C_host, layout, oneapi::mkl::transpose::nontrans, N, N, N);

    std::memcpy(C_host_ref, C_host, sizeof(float)*N*N);
    gpuErrchk(hipMemcpy((void*)A_dev, (void*)A_host, sizeof(float)*N*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void*)B_dev, (void*)B_host, sizeof(float)*N*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void*)C_dev, (void*)C_host, sizeof(float)*N*N,  hipMemcpyHostToDevice));
    gpuErrchk(hipDeviceSynchronize());

  }

  void run(std::vector<cl::sycl::event>& events) {
    //hipSetDevice(0);
    float alpha = 1.0;
    cublasErrchk(rocblas_sgemm(handle, rocblas_operation_none,rocblas_operation_none,
                        N,N,N,
                        &alpha,A_dev, N,
                        B_dev, N, &alpha,
                        C_dev, N));
    gpuErrchk(hipDeviceSynchronize());
  }

  bool verify(VerificationSetting &ver) {
    //gpuErrchk(hipMemcpy((void*)y_host, (void*)y_dev, sizeof(float)*N,    hipMemcpyDeviceToHost));
    //gpuErrchk(hipMemcpy((void*)x_host, (void*)x_dev, sizeof(float)*N,    hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy((void*)C_host, (void*)C_dev, sizeof(float)*N*N,  hipMemcpyDeviceToHost));    
    int m = N;
    int n = N;
    float alpha = 1;
    int incx = 1;
    int size = N*N;
    gemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
           &N, &N, &N,
           &alpha, A_host, &N,
           B_host, &N, &alpha,
           C_host_ref, &N);
    //args.device_queue.wait();
    return check_equal_vector(C_host, C_host_ref, size, incx, size, std::cout);
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

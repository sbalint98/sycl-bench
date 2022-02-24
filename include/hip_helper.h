#include <rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
  
    inline const char *rocblas_error_map(rocblas_status error) {
        switch (error) {
            case rocblas_status_success: return "rocblas_status_success";
            case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
            case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
            case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
            case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
            case rocblas_status_memory_error: return "rocblas_status_memory_error";
            case rocblas_status_internal_error: return "rocblas_status_internal_error";
            case rocblas_status_perf_degraded: return "rocblas_status_perf_degraded";
            case rocblas_status_size_query_mismatch: return "rocblas_status_size_query_mismatch";
            case rocblas_status_size_increased: return "rocblas_status_size_increased";
            case rocblas_status_size_unchanged: return "rocblas_status_size_unchanged";
            case rocblas_status_invalid_value: return "rocblas_status_invalid_value";
            case rocblas_status_continue: return "rocblas_status_continue";
            case rocblas_status_check_numerics_fail: return "rocblas_status_check_numerics_fail";

            default: return "<unknown>";
        }
    }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %d\n", file, line);
      if (abort) exit(code);
   }
}

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(rocblas_status code, const char *file, int line, bool abort=true)
{
   if (code != rocblas_status_success) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n",  rocblas_error_map(code), file, line);
      if (abort) exit(code);
   }
}

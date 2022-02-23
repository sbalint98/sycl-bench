#include "common.h"
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
  T* x_device;
  T* x_host;
  int64_t* result_p_host;
  int64_t* result_p_device;


public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    sycl::context ctx = args.device_queue.get_context();
    sycl::device dev = args.device_queue.get_device();
    // host memory intilization
    x_host = (T*)sycl::malloc_host(sizeof(T)*args.problem_size, ctx);
    x_device = (T*)sycl::malloc_device(sizeof(T)*args.problem_size, dev, ctx);
    result_p_host = (int64_t*)sycl::malloc_host(sizeof(int64_t), ctx);
    result_p_device = (int64_t*)sycl::malloc_device(sizeof(int64_t), dev, ctx);
    for (size_t i =0; i < args.problem_size; i++) {
      x_host[i] = static_cast<T>(i);
    }
    args.device_queue.memcpy(x_device, x_host, sizeof(T)*args.problem_size);
    args.device_queue.memcpy(result_p_device, result_p_host, sizeof(int64_t));
  }

  void run(std::vector<cl::sycl::event>& events) {
    std::vector<cl::sycl::event> asdf=  std::vector<cl::sycl::event>();
    sycl::event ev = oneapi::mkl::blas::column_major::iamax(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas> {args.device_queue}, args.problem_size, x_device, 1,
                                                              result_p_device);
    events.push_back(ev);

  }

  bool verify(VerificationSetting &ver) {
    //Triggers writeback
    // output_buf.reset();

    // bool pass = true;
    // for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
    //     auto expected = input1[i] + input2[i];
    //     if(expected != output[i]){
    //         pass = false;
    //         break;
    //     }
    //   }    
    // return pass;
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
  app.run<VecAddBench<double>>();
  return 0;
}

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

#include <eikonal.h>
#include <tracer.h>
#include <volume.h>
#include <enoki/autodiff.h>
#include <enoki/special.h>

using namespace enoki;
using namespace drrt;

using myFloat = Float<true, true>;
using myVec3 = Vector3f<true, true>;
using myVec2 = Vector2f<true, true>;
using myMask = mask_t<myFloat>;

void test_tracer(int nres, int nrays, float ds) {
  float h = 1;

  ScalarVector3i res(nres, nres, nres);
  myFloat rif = zero<myFloat>(nres*nres*nres) + 1;
  myFloat x = linspace<myFloat>(0.0f, nres, nrays);
  myFloat y = linspace<myFloat>(0.0f, nres, nrays);
  myVec2 grid = meshgrid<myFloat>(x, y);

  myVec3 pos = myVec3(grid.x(), grid.y(), 0);
  myVec3 vel = myVec3(zero<myFloat>(nrays*nrays),
                      zero<myFloat>(nrays*nrays),
                      zero<myFloat>(nrays*nrays)+1);


  set_requires_gradient(rif);
  myFloat::set_graph_simplification_(false);

  Tracer<true, true> trace = Tracer<true, true>();
  auto [xt, vt] = trace.trace(rif, res, pos, vel, h, ds);

  auto loss = hsum(xt);

  backward(loss);

  gradient(rif);
  //std::cout << gradient(rif) << std::endl;
}

void test_autodiff() {
  FloatD a = 1.f;
  set_requires_gradient(a);

  FloatD b = erf(a);
  set_label(a, "a");
  set_label(b, "b");

  backward(b);
  std::cout << gradient(a) << std::endl;
}

void test_volume() {

  ScalarVector3i res(3, 3, 3);
  ScalarVector3i res2(5, 5, 5);
  myFloat rif = zero<myFloat>(27) + 1;
  myFloat rif2 = zero<myFloat>(125) + 1;

  volume<false, false> vol(res, rif, 0.5);
  volume<false, false> vol2(res2, rif2, 0.25);

  myVec3 p1(0.5, 0.5, 0.5);
  auto [n, nx] = vol.eval_grad(p1, true);
  auto [n2, nx2] = vol2.eval_grad(p1, true);
  std::cout << "res: 3"
    << n << std::endl 
    << nx << std::endl;
  std::cout << "res: 5"
    << n2 << std::endl 
    << nx2 << std::endl;
}

void test_cylinder() { 
  using myFloat = Float<false, true>;
  using myVec3 = Vector3f<false, true>;
  using myVec2 = Vector2f<false, true>;
  using myMask = mask_t<myFloat>;

  myFloat rif = zero<myFloat>(8) + 1;
  cylinder_volume<false, true> vol(rif, 2.0, 4.0);

  myVec3 p1(1.5, 0.5, 1.0);
  myVec3 v1(0.0, 1.0, 0.0);
  myVec3 sp(2.0, 2.0, 2.0);

  auto [n, nx] = vol.eval_grad(p1, true);

  std::cout << "vol before splat: "
            << vol.get_data()
            << std::endl;

  vol.splat(p1, 3.0, v1);
  std::cout << "vol after splat: "
            << vol.get_data()
            << std::endl;

  Tracer<false, true> trace = Tracer<false, true>();
  auto [xt, vt, dist2] = trace.trace_cable(rif, 2.0, 4.0, p1, v1, sp, 0.01);

  std::cout << "rays:" << std::endl
            << xt << std::endl
            << vt << std::endl;



}

void compare_back(int nres, int nrays, float ds) {
  using myFloat = Float<false, true>;
  using myVec3 = Vector3f<false, true>;
  using myVec2 = Vector2f<false, true>;
  using myMask = mask_t<myFloat>;

  float h = 1;

  ScalarVector3i res(nres, nres, nres);
  myFloat rif = zero<myFloat>(nres*nres*nres) + 1;
  myFloat x = linspace<myFloat>(0.0f, nres, nrays);
  myFloat y = linspace<myFloat>(0.0f, nres, nrays);
  myVec2 grid = meshgrid<myFloat>(x, y);

  myVec3 pos = myVec3(grid.x(), grid.y(), 0);
  myVec3 vel = myVec3(zero<myFloat>(nrays*nrays),
                      zero<myFloat>(nrays*nrays),
                      zero<myFloat>(nrays*nrays)+1);


  //set_requires_gradient(rif);

  Tracer<false, true> trace = Tracer<false, true>();
  auto [xt, vt] = trace.trace(rif, res, pos, vel, h, ds);

  auto ones = zero<myVec3>(nrays * nrays) + 1;
  auto grif = trace.backtrace(rif, res, xt, vt, ones,
                              ones, h, ds);

}

void profile_stepsize() {
  //test_volume();
  //test_cylinder();
  // test_autodiff();
  int num_tests = 3;
  int nres = 33;
  int nrays = 512;
  float ds = 0.1;
  std::vector<float> step_sizes = {0.3, 0.3, 0.33, 0.33, 0.37, 0.37, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  //std::vector<float> step_sizes;// = {1.0, 0.8, 0.6, 0.4};
  //for (int i = 1; i <= 10; ++i) {
  //  step_sizes.push_back(i*ds + 0.2);
  //}

  size_t free_mem, total_mem;

  std::vector<float> ad_times;
  std::vector<size_t> ad_mems;
  for (int i = 0; i < step_sizes.size(); ++i) {
    ds = step_sizes[i];
    auto start_time = std::chrono::system_clock::now();
    test_tracer(nres, nrays, ds);
    cuda_mem_get_info(&free_mem, &total_mem);
    auto ad_time = std::chrono::system_clock::now();
    float ad_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ad_time - start_time).count();
    ad_times.push_back(ad_time_ms);
    ad_mems.push_back(total_mem - free_mem);
    std::cout << "iter " << i << std::endl;
    cuda_malloc_trim();
  }

  cuda_malloc_trim();
  compare_back(nres, nrays, 0.1);
  cuda_mem_get_info(&free_mem, &total_mem);
  cuda_malloc_trim();

  std::vector<float> back_times;
  std::vector<size_t> back_mems;
  for (int i = 0; i < step_sizes.size(); ++i) {
    ds = step_sizes[i];
    auto start_time = std::chrono::system_clock::now();
    compare_back(nres, nrays, ds);
    cuda_mem_get_info(&free_mem, &total_mem);
    auto ba_time = std::chrono::system_clock::now();
    float ba_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ba_time - start_time).count();
    back_times.push_back(ba_time_ms);
    back_mems.push_back(total_mem - free_mem);
    std::cout << "iter " << i << std::endl;
    cuda_malloc_trim();
  }

  std::cout << "ds ad_times back_time ad_mem back_mem" << std::endl;
  for (int i = 0; i < step_sizes.size(); ++i) {
    std::cout << 1.0 / step_sizes[i] << ' ';
    std::cout << ad_times[i] / 1000.0 << ' ';
    std::cout << back_times[i] / 1000.0 << ' ';
    std::cout << ad_mems[i] / 1024.0 / 1024.0 / 1024.0 - 0.996 << ' ';
    std::cout << back_mems[i] / 1024.0 / 1024.0 / 1024.0 - 0.996 << std::endl;
  }

  //std::cout << "ds: ";
  //for (float ds : step_sizes) std::cout << ds << ',';
  //std::cout << std::endl;

  //std::cout << "AD:" << std::endl;
  //std::cout << "time(ms): [";
  //for (float t : ad_times) {
  //  std::cout << ' ' << t << ',';
  //}
  //std::cout << ']' << std::endl;

  //std::cout << "memory(MB): [";
  //for (size_t t : ad_mems) {
  //  float memMB = (t / 1024.0 / 1024.0) - 996;
  //  std::cout << ' ' << memMB << ",";
  //}
  //std::cout << ']' << std::endl;

  //std::cout << "BA:" << std::endl;
  //std::cout << "time(ms): [";
  //for (float t : back_times) {
  //  std::cout << t << ',';
  //}
  //std::cout << ']' << std::endl;

  //std::cout << "memory(MB): [";
  //for (size_t t : back_mems) {
  //  float memMB = (t / 1024.0 / 1024.0) - 996;
  //  std::cout << ' ' << memMB << ",";
  //}
  //std::cout << ']' << std::endl;
}

void profile_resolution() {
  // TODO(ateh): same as stepsize, but change the resolution of the volume to see the differences
  // TODO: should also do number of rays
  int num_tests = 3;
  int nres = 3;
  int nrays = 256;
  float ds = 0.5;
  std::vector<int> res_sizes = {3, 3, 5, 9, 17, 33, 65, 129, 257};

  size_t free_mem, total_mem;

  std::vector<float> back_times;
  std::vector<size_t> back_mems;
  for (int i = 0; i < res_sizes.size(); ++i) {
    nres = res_sizes[i];
    auto start_time = std::chrono::system_clock::now();
    compare_back(nres, nrays, ds);
    cuda_mem_get_info(&free_mem, &total_mem);
    auto ba_time = std::chrono::system_clock::now();
    float ba_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ba_time - start_time).count();
    back_times.push_back(ba_time_ms);
    back_mems.push_back(total_mem - free_mem);
    std::cout << "iter " << i << std::endl;
    cuda_malloc_trim();
  }

  std::cout << "BA:" << std::endl;
  std::cout << "time(ms): [";
  for (float t : back_times) {
    std::cout << ' ' << t << ',';
  }
  std::cout << ']' << std::endl; 

  std::cout << "memory(MB): [";
  for (size_t t : back_mems) {
    float memMB = (t / 1024.0 / 1024.0) - 996;
    std::cout << ' ' << memMB << ",";
  }
  std::cout << ']' << std::endl;

  std::vector<float> ad_times;
  std::vector<size_t> ad_mems;
  for (int i = 0; i < res_sizes.size(); ++i) {
    nres = res_sizes[i];
    auto start_time = std::chrono::system_clock::now();
    test_tracer(nres, nrays, ds);
    cuda_mem_get_info(&free_mem, &total_mem);
    auto ad_time = std::chrono::system_clock::now();
    float ad_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ad_time - start_time).count();
    ad_times.push_back(ad_time_ms);
    ad_mems.push_back(total_mem - free_mem);
    std::cout << "iter " << i << std::endl;
    cuda_malloc_trim();
  }

  std::cout << "nres: [";
  for (int nr : res_sizes) std::cout << nr << ',';
  std::cout << "]" << std::endl;

  std::cout << "AD:" << std::endl;
  std::cout << "time(ms): [";
  for (float t : ad_times) {
    std::cout << ' ' << t << ',';
  }
  std::cout << ']' << std::endl; 

  std::cout << "memory(MB): [";
  for (size_t t : ad_mems) {
    float memMB = (t / 1024.0 / 1024.0) - 996;
    std::cout << ' ' << memMB << ",";
  }
  std::cout << ']' << std::endl;

  //compare_back(nres, nrays, 0.1);
  //cuda_mem_get_info(&free_mem, &total_mem);
  //cuda_malloc_trim();

}

int main() { 
  std::cout << "step size -----" << std::endl;
  profile_stepsize(); 
  //std::cout << "finiehs step size" << std::endl << std::endl;
  //std::cout << "resolution -----" << std::endl;
  //profile_resolution();
}

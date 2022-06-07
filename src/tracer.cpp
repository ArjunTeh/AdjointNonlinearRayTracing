#include "tracer.h"

#include "eikonal.h"
#include "volume.h"

#include <iostream>
#include <enoki/cuda.h>
#include <enoki/array.h>
#include <enoki/matrix.h>
#include <algorithm>

using namespace enoki;

namespace drrt {

template <bool ad, bool gpu>
void Tracer<ad, gpu>::test_in(Vector3f<ad, gpu> p) {
  using Vec = Vector3f<ad, gpu>;
  using Ivec = Vector3i<ad, gpu>;
  FloatC wow = FloatC(0.5);
  Vec one = Vec(1.1, 1.1, 1.1) + wow;
  Ivec fl = floor2int<Ivec, Vec>(one);
  std::cout << one << std::endl;
  std::cout << fl << std::endl;
  return;
}

template <bool ad, bool gpu>
Vector3fC Tracer<ad, gpu>::tester() {
  Vector3fC a = Vector3fC(0, 0, 0);
  // std::cout << a << std::endl;
  return a;
}

template<bool ad, bool gpu>
std::pair<Vector3f<ad, gpu>, Vector3f<ad, gpu>>
Tracer<ad, gpu>::trace(Float<ad, gpu>& rif,
                       ScalarVector3i res,
                       Vector3f<ad, gpu>& pos,
                       Vector3f<ad, gpu>& vel,
                       scalar_t<Float<ad, gpu>> h,
                       scalar_t<Float<ad, gpu>> delta_s) {
  // TODO(ateh): bad syntax - maybe just pick version
  using Mask = mask_t<Float<ad, gpu>>;
  using fVector3 = Vector3f<ad, gpu>;

  // generate volume
  volume<ad, gpu> grid = volume<ad, gpu>(res, rif, h);

  // intialize the integrator
  int max_steps = 4 * h * hmax(res) / delta_s;

  fVector3 x(pos);
  fVector3 v(vel);

  fVector3 xt(pos);
  fVector3 vt(vel);

  Float<ad, gpu> ds(delta_s);

  auto inside = grid.inbounds(x);
  auto escaped = inside & !inside;
  auto active = !escaped;

  int i;
  for (i=0; i < max_steps; ++i) {
    // step forward
    auto [n, nx] = grid.eval_grad(x, inside);

    v = fmadd(ds * n, nx, v);
    x = fmadd(ds, v, x);

    Mask cur_inside = grid.inbounds(x);
    Mask cross = inside & (!cur_inside);
    escaped |= cross;
    escaped |= grid.escaped(x, v);
    active &= !escaped;

    xt[cross] = x;
    vt[cross] = v;

    if (all(escaped)) {
      break;
    }

    inside = cur_inside;
  }

  if (any(active)) {
    std::cout << "failed to exit all rays" << std::endl;
    //Vector3f<false, gpu> x_failed = detach(gather<fVector3>(x, arange<Int<ad, gpu>>(slices(x)), active));
    //std::cout << x_failed << std::endl;
    //auto x_detach = detach(x);
    //auto act_det = detach(active);
    xt[!escaped] = x;
  }

  // trace until we get the exit ray
  return std::make_pair(xt, vt);
}

template <bool ad, bool gpu>
std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Bool<ad, gpu>>
Tracer<ad, gpu>::trace_plane(Float<ad, gpu>& rif,
                             ScalarVector3i res,
                             Vector3f<ad, gpu>& pos,
                             Vector3f<ad, gpu>& vel,
                             Vector3f<ad, gpu>& pln_o,
                             Vector3f<ad, gpu>& pln_d,
                             scalar_t<Float<ad, gpu>> h,
                             scalar_t<Float<ad, gpu>> delta_s) {
  // TODO(ateh): bad syntax - maybe just pick version
  using Mask = mask_t<Float<ad, gpu>>;
  using fVector3 = Vector3f<ad, gpu>;

  // generate volume
  volume<ad, gpu> grid = volume<ad, gpu>(res, rif, h);

  // intialize the integrator
  int max_steps = 4 * h * hmax(res) / delta_s;

  fVector3 x(pos);
  fVector3 v(vel);

  fVector3 xt(pos);
  fVector3 vt(vel);

  Float<ad, gpu> ds(delta_s);

  auto inside = grid.inbounds(x);
  auto escaped = inside & !inside;
  auto active = !escaped;

  int i;
  for (i=0; i < max_steps; ++i) {
    // step forward
    auto [n, nx] = grid.eval_grad(x, inside);

    v = fmadd(ds * n, nx, v);
    x = fmadd(ds, v, x);

    //std::cout << x << std::endl;

    Mask past_pln = dot(x - pln_o, pln_d) > 0;
    Mask cur_inside = grid.inbounds(x) & !past_pln;
    Mask cross = inside & (!cur_inside);
    escaped |= cross;
    escaped |= grid.escaped(x, v);
    active &= !escaped;

    xt[cross] = x;
    vt[cross] = v;

    if (all(escaped)) {
      break;
    }

    inside = cur_inside;
  }

  if (any(active)) {
    std::cout << "failed to exit all rays" << std::endl;
    //Vector3f<false, gpu> x_failed = detach(gather<fVector3>(x, arange<Int<ad, gpu>>(slices(x)), active));
    //std::cout << x_failed << std::endl;
    //auto x_detach = detach(x);
    //auto act_det = detach(active);
    xt[!escaped] = x;
  }

  // trace until we get the exit ray
  return std::make_tuple(xt, vt, Bool<ad, gpu>(!escaped));
}

template <bool ad, bool gpu>
std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Float<ad, gpu>>
Tracer<ad, gpu>::trace_target(Float<ad, gpu>& rif, 
                              ScalarVector3i res,
                              Vector3f<ad, gpu>& pos, 
                              Vector3f<ad, gpu>& vel,
                              Vector3f<ad, gpu>& target,
                              scalar_t<Float<ad, gpu>> h,
                              scalar_t<Float<ad, gpu>> delta_s) {

  // TODO(ateh): bad syntax - maybe just pick version
  using Mask = mask_t<Float<ad, gpu>>;
  using fVector3 = Vector3f<ad, gpu>;

  // generate volume
  volume<ad, gpu> grid = volume<ad, gpu>(res, rif, h);

  // intialize the integrator
  int max_steps = 4 * h * hmax(res) / delta_s;

  fVector3 x(pos);
  fVector3 v(vel);

  fVector3 xt(pos);
  fVector3 vt(vel);

  Float<ad, gpu> dist2 = squared_norm(x - target);

  Float<ad, gpu> ds(delta_s);

  auto inside = grid.inbounds(x);
  auto escaped = inside & !inside;
  auto active = !escaped;

  int i;
  for (i=0; i < max_steps; ++i) {
    // step forward
    auto [n, nx] = grid.eval_grad(x, inside);

    v = fmadd(ds * n, nx, v);
    x = fmadd(ds, v, x);

    Float<ad, gpu> cur_dist2 = squared_norm(x - target);
    Mask closer = cur_dist2 < dist2;

    Mask cur_inside = grid.inbounds(x);
    Mask cross = inside & (!cur_inside);
    escaped |= cross;
    escaped |= grid.escaped(x, v);
    active &= !escaped;

    xt[closer] = x;
    vt[closer] = v;
    dist2[closer] = cur_dist2;

    if (all(escaped)) {
      break;
    }

    inside = cur_inside;
  }

  if (any(active)) {
    std::cout << "failed to exit all rays" << std::endl;
    //xt[!escaped] = x;
  }

  return std::make_tuple(xt, vt, dist2);
}

template <bool ad, bool gpu>
std::pair<Vector3f<ad, gpu>, Vector3f<ad, gpu>> 
Tracer<ad, gpu>::trace_sdf(Float<ad, gpu>& rif, 
                           Float<ad, gpu>& sdf, 
                           ScalarVector3i res, 
                           Vector3f<ad, gpu>& pos, 
                           Vector3f<ad, gpu>& vel, 
                           scalar_t<Float<ad, gpu>> h, 
                           scalar_t<Float<ad, gpu>> delta_s) {

  using Mask = mask_t<Float<ad, gpu>>;
  using fVector3 = Vector3f<ad, gpu>;

  // generate volume
  volume<ad, gpu> grid = volume<ad, gpu>(res, rif, h);
  volume<ad, gpu> sdf_vol = volume<ad, gpu>(res, sdf, h);

  // intialize the integrator
  int max_steps = 2 * h * hmax(res) / delta_s;

  fVector3 x(pos);
  fVector3 v(vel);

  fVector3 xt(pos);
  fVector3 vt(vel);

  Float<ad, gpu> ds(delta_s);

  auto inside = grid.inbounds(x);
  auto escaped = inside & !inside;
  auto active = !escaped;

  auto [dist, distx] = sdf_vol.eval_grad(x, active);
  active = dist < 0;

  int i;
  for (i=0; i < max_steps; ++i) {
    // step forward
    auto [n, nx] = grid.eval_grad(x, inside);

    v = fmadd(ds * n, nx, v);
    x = fmadd(ds, v, x);

    auto [dist, distx] = sdf_vol.eval_grad(x, inside);
    Mask cur_inside = dist < 0;
    Mask cross = inside & (!cur_inside);
    escaped |= cross;
    escaped |= grid.escaped(x, v);
    active &= !escaped;

    xt[cross] = x;
    vt[cross] = v;

    if (all(escaped)) {
      break;
    }

    inside = cur_inside;
  }

  if (i == max_steps) {
    std::cout << "failed to exit all rays" << std::endl;
  }

  // trace until we get the exit ray
  return std::make_pair(xt, vt);
}

template <bool ad, bool gpu>
std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Float<ad, gpu>>
Tracer<ad, gpu>::trace_cable(Float<ad, gpu>& rif, 
                             scalar_t<Float<ad, gpu>> radius,
                             scalar_t<Float<ad, gpu>> length, 
                             Vector3f<ad, gpu>& pos, 
                             Vector3f<ad, gpu>& vel, 
                             Vector3f<ad, gpu>& target,
                             scalar_t<Float<ad, gpu>> ds) {

  // TODO(ateh): bad syntax - maybe just pick version
  using Mask = mask_t<Float<ad, gpu>>;
  using fVector3 = Vector3f<ad, gpu>;


  // generate volume
  cylinder_volume<ad, gpu> cable = cylinder_volume<ad, gpu>(rif, radius, length);


  // intialize the integrator
  int max_steps = int(4 * length / ds);

  fVector3 x(pos);
  fVector3 v(vel);

  fVector3 xt(pos);
  fVector3 vt(vel);

  Float<ad, gpu> dist2 = squared_norm(x - target);
  
  //Float<ad, gpu> ds(delta_s);

  auto inside = cable.inbounds(x);
  auto escaped = inside & !inside;
  auto active = !escaped;

  int i;
  for (i=0; i < max_steps; ++i) {
    // step forward
    auto [n, nx] = cable.eval_grad(x, inside);

    v[active] = fmadd(ds * n, nx, v);
    x[active] = fmadd(ds, v, x);

    Float<ad, gpu> cur_dist2 = squared_norm(x - target);
    Mask closer = cur_dist2 < dist2;

    Mask cur_inside = cable.inbounds(x);
    Mask cross = inside & (!cur_inside);
    escaped |= cross;
    escaped |= cable.escaped(x, v);
    active &= !escaped;

    xt[closer] = x;
    vt[closer] = v;
    dist2[closer] = cur_dist2;

    if (all(escaped)) {
      break;
    }

    inside = cur_inside;
  }

  if (any(active)) {
    std::cout << "failed to exit all rays" << std::endl;
    //xt[!escaped] = x;
  }

  return std::make_tuple(xt, vt, dist2);
}

template <bool ad, bool gpu>
Float<false, gpu>
Tracer<ad, gpu>::backtrace(Float<false, gpu>& rif,
                           ScalarVector3i res,
                           Vector3f<false, gpu>& xt,
                           Vector3f<false, gpu>& vt,
                           Vector3f<false, gpu>& dx,
                           Vector3f<false, gpu>& dv,
                           scalar_t<Float<false, gpu>> h,
                           scalar_t<Float<false, gpu>> ds) {

  using Mask = mask_t<Float<false, gpu>>;
  using fVector3 = Vector3f<false, gpu>;
  using myFloat = Float<false, gpu>;
  using Matrix3 = Matrix<myFloat, 3>;

  // generate volume
  myFloat zeros = zero<myFloat>(slices(rif));
  volume<false, gpu> grid = volume<false, gpu>(res, rif, h);
  volume<false, gpu> grad = volume<false, gpu>(res, zeros, h);

  // ignore the boundary term? maybe just let it go for whatever amount
  fVector3 x(xt);
  fVector3 v(vt);

  fVector3 la(dx);
  fVector3 mu(dv + ds*dx);
  //fVector3 mu(dv);

  Mask escaped = grid.escaped(x, -v);
  Mask active = !escaped;

  auto [n, nx] = grid.eval_grad(x, active);
  int max_steps = 2*h*hmax(res) / ds;
  int i;
  for (i=0; i<max_steps; ++i) {
    x = fmadd(-ds, v, x);
    std::tie(n, nx) = grid.eval_grad(x, active);
    Matrix3 Hess = grid.eval_hess(x, active);
    v = fmadd(-ds*n, nx, v);

    active &= !grid.escaped(x, -v);
    if (none(active)) {
      break;
    }

    myFloat dn = dot(mu, nx);
    fVector3 dnx = n*mu;
    grad.splat(x, dn*ds, dnx*ds, active);

    la = la + ds * (dn * nx + n * Hess * mu);
    mu = mu + ds * la;
  }

  auto val = grad.get_data();
  return val;
}


template <bool ad, bool gpu>
Float<false, gpu>
Tracer<ad, gpu>::backtrace_sdf(Float<false, gpu>& rif,
                               Float<false, gpu>& sdf,
                               ScalarVector3i res,
                               Vector3f<false, gpu>& xt,
                               Vector3f<false, gpu>& vt,
                               Vector3f<false, gpu>& dx,
                               Vector3f<false, gpu>& dv,
                               scalar_t<Float<false, gpu>> h,
                               scalar_t<Float<false, gpu>> ds) {

  using Mask = mask_t<Float<false, gpu>>;
  using fVector3 = Vector3f<false, gpu>;
  using myFloat = Float<false, gpu>;
  using Matrix3 = Matrix<myFloat, 3>;

  // generate volume
  myFloat zeros = zero<myFloat>(slices(rif));
  volume<false, gpu> grid = volume<false, gpu>(res, rif, h);
  volume<false, gpu> grad = volume<false, gpu>(res, zeros, h);
  volume<false, gpu> sdf_vol = volume<false, gpu>(res, sdf, h);

  // ignore the boundary term? maybe just let it go for whatever amount
  fVector3 x(xt);
  fVector3 v(vt);

  fVector3 la(dx);
  fVector3 mu(dv + ds*dx);
  //fVector3 mu(dv);

  Mask escaped = grid.escaped(x, -v);
  Mask active = !escaped;
  auto [dist, distx] = sdf_vol.eval_grad(x, active);
  Mask outside = dist >= 0;

  auto [n, nx] = grid.eval_grad(x, active);
  int max_steps = 2*h*hmax(res) / ds;
  int i;
  for (i=0; i<max_steps; ++i) {
    x = fmadd(-ds, v, x);
    std::tie(n, nx) = grid.eval_grad(x, active);
    Matrix3 Hess = grid.eval_hess(x, active);
    v = fmadd(-ds*n, nx, v);

    std::tie(dist, distx) = sdf_vol.eval_grad(x, active);
    
    // TODO(ateh): the cross condition should be debugged
    active &= !grid.escaped(x, -v);
    Mask cross = !outside & (dist >= 0);
    active &= !cross;
    if (none(active)) {
      break;
    }
    outside = dist >=0;

    myFloat dn = dot(mu, nx);
    fVector3 dnx = n*mu;
    grad.splat(x, dn*ds, dnx*ds, active);

    la = la + ds * (dn * nx + n * Hess * mu);
    mu = mu + ds * la;
  }

  auto val = grad.get_data();
  return val;
}

template <bool ad, bool gpu>
Float<false, gpu> 
Tracer<ad, gpu>::backtrace_cable(Float<false, gpu>& rif, 
                                 scalar_t<Float<false, gpu>> radius, 
                                 scalar_t<Float<false, gpu>> length, 
                                 Vector3f<false, gpu>& xt, 
                                 Vector3f<false, gpu>& vt, 
                                 Vector3f<false, gpu>& dx, 
                                 Vector3f<false, gpu>& dv, 
                                 scalar_t<Float<ad, gpu>> ds) {

  using Mask = mask_t<Float<false, gpu>>;
  using fVector3 = Vector3f<false, gpu>;
  using myFloat = Float<false, gpu>;
  using Matrix3 = Matrix<myFloat, 3>;

  // generate volume
  myFloat zeros = zero<myFloat>(slices(rif));
  cylinder_volume<false, gpu> cable = cylinder_volume<false, gpu>(rif, radius, length);
  cylinder_volume<false, gpu> grad = cylinder_volume<false, gpu>(zeros, radius, length);

  // ignore the boundary term? maybe just let it go for whatever amount
  fVector3 x(xt);
  fVector3 v(vt);

  fVector3 la(dx);
  fVector3 mu(dv + ds*dx);
  //fVector3 mu(dv);

  Mask escaped = cable.escaped(x, -v);
  Mask active = !escaped;

  auto [n, nx] = cable.eval_grad(x, active);
  int max_steps = int(4*length / ds);
  int i;
  for (i=0; i<max_steps; ++i) {
    x = fmadd(-ds, v, x);
    std::tie(n, nx) = cable.eval_grad(x, active);
    Matrix3 Hess = cable.eval_hess(x, active);
    v = fmadd(-ds*n, nx, v);

    active &= !cable.escaped(x, -v);
    if (none(active)) {
      break;
    }

    myFloat dn = dot(mu, nx);
    fVector3 dnx = n*mu;
    grad.splat(x, dn*ds, dnx*ds, active);

    la = la + ds * (dn * nx + n * Hess * mu);
    mu = mu + ds * la;
  }

  auto val = grad.get_data();
  return val;
}

template class Tracer<true, false>;
template class Tracer<false, false>;
template class Tracer<true, true>;
template class Tracer<false, true>;

// template std::pair<Vector3fD, Vector3fD> trace<true>(FloatD& rif,
//                                                      ScalarVector3i res,
//                                                      Vector3fD& pos,
//                                                      Vector3fD& vel,
//                                                      float h,
//                                                      float ds);

// template std::pair<Vector3fC, Vector3fC> trace<false>(FloatC& rif,
//                                                       ScalarVector3i res,
//                                                       Vector3fC& pos,
//                                                       Vector3fC& vel,
//                                                       float h,
//                                                       float ds);

} // namespace drrt

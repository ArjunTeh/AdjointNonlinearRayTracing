#include "volume.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/cuda.h>
#include <enoki/matrix.h>

using namespace enoki;

namespace drrt {

constexpr float FLOAT_EPSILON = 1e-6;

template <bool ad, bool gpu>
cylinder_volume<ad, gpu>::cylinder_volume() : data_(0.0), radius_(0), length_(0) {}

template <bool ad, bool gpu>
cylinder_volume<ad, gpu>::cylinder_volume(const Float<ad, gpu>& data,
                                          scalar_t<Float<ad, gpu>> radius,
                                          scalar_t<Float<ad, gpu>> length) 
    : data_(data), radius_(radius), length_(length) {}

template <bool ad, bool gpu>
std::pair<Float<ad, gpu>, Vector3f<ad, gpu>> cylinder_volume<ad, gpu>::eval_grad(
    Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const {

  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;
  using fMatrix3 = Matrix<scalar_t<myFloat>, 3>;

  // technically the y-origin is length_/2, but we ignore anyway
  fVector3 xs = p - radius_;
  xs[1] = 0;

  size_t res = slices(data_);
  myFloat r = norm(xs);
  scalar_t<myFloat> h = radius_ / (res-1);

  myFloat rm = r / h;
  myInt idx0 = clamp(floor2int<myInt, myFloat>(rm), 0, res - 1);
  myInt idx1 = clamp(idx0 + 1, 0, res - 1);

  myFloat w0 = rm - myFloat(idx0), w1 = 1.0f - w0;

  myFloat val0 = gather<myFloat>(data_, idx0);
  myFloat val1 = gather<myFloat>(data_, idx1);

  myFloat f = val0*w1 + val1*w0;
  myFloat rx = (val1 - val0) / h;
  fVector3 fx = rx * normalize(xs);
  fx[r < FLOAT_EPSILON] = 0;

  return std::make_pair(f, fx);
}

template <bool ad, bool gpu>
Matrix<Float<ad, gpu>, 3> cylinder_volume<ad, gpu>::eval_hess(
    Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const {

  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;
  using myMatrix = Matrix<myFloat, 3>;

  // technically the y-origin is length_/2, but we ignore anyway
  fVector3 xs = p - radius_;
  xs[1] = 0;

  size_t res = slices(data_);
  myFloat r = norm(xs);
  scalar_t<myFloat> h = radius_ / (res-1);

  myFloat rm = r / h;
  myInt idx0 = clamp(floor2int<myInt, myFloat>(rm), 0, res - 1);
  myInt idx1 = clamp(idx0 + 1, 0, res - 1);

  myFloat w0 = rm - myFloat(idx0), w1 = 1.0f - w0;

  myFloat val0 = gather<myFloat>(data_, idx0);
  myFloat val1 = gather<myFloat>(data_, idx1);

  myFloat rx = (val1 - val0) / h;

  fVector3 xhat = normalize(xs);
  xhat[r < FLOAT_EPSILON] = 0;
  myMatrix H(0);
  set_slices(H, slices(p));

  // since the projection is the y plane,
  // we ignore all of the y components
  H(0, 0) = 1 - (xhat[0] * xhat[0]);
  //H(0, 1) = -(xhat[0] * xhat[1]);
  H(0, 2) = -(xhat[0] * xhat[2]);
  H(1, 0) = 0;
  H(1, 1) = 0;
  H(1, 2) = 0;
  H(2, 0) = -(xhat[2] * xhat[0]);
  //H(2, 1) = -(xhat[2] * xhat[1]);
  H(2, 2) = 1 - (xhat[2] * xhat[2]);

  H = H * (rx / r);
  H[r < FLOAT_EPSILON] = myMatrix(0.0);

  return H;
}

template <bool ad, bool gpu>
void cylinder_volume<ad, gpu>::splat(Vector3f<ad, gpu> const& pos,
                                   Float<ad, gpu> const& val,
                                   Vector3f<ad, gpu> const& grad, Mask active) {
  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;
  using myMatrix = Matrix<myFloat, 3>;

  // technically the y-origin is length_/2, but we ignore anyway
  fVector3 xs = pos - radius_;
  xs[1] = 0;

  size_t res = slices(data_);
  myFloat r = norm(xs);
  fVector3 rx = normalize(xs);
  scalar_t<myFloat> h = radius_ / (res-1);

  myFloat rm = r / h;
  myInt idx0 = clamp(floor2int<myInt, myFloat>(rm), 0, res - 1);
  myInt idx1 = clamp(idx0 + 1, 0, res - 1);

  myFloat w0 = rm - myFloat(idx0), w1 = 1.0f - w0;

  // splat value
  scatter_add(data_, val*w1, idx0, active);
  scatter_add(data_, val*w0, idx1, active);

  myFloat grad_val = dot(grad, rx);
  grad_val[r < FLOAT_EPSILON] = 0;//norm(grad);

  // splat gradient
  scatter_add(data_, -grad_val / h, idx0, active);
  scatter_add(data_, grad_val / h, idx1, active);
}

template <bool ad, bool gpu>
mask_t<Float<ad, gpu>> cylinder_volume<ad, gpu>::inbounds(Vector3f<ad, gpu> p) const {
  Vector3f<ad, gpu> pl = p - radius_;
  Float<ad, gpu> r = (pl.x()*pl.x() + pl.z()*pl.z());
  auto inlength = (p.y() < length_) & (p.y() >= 0);
  return (r < (radius_*radius_)) & inlength;
}

template <bool ad, bool gpu>
mask_t<Float<ad, gpu>> cylinder_volume<ad, gpu>::escaped(Vector3f<ad, gpu> p,
                                                       Vector3f<ad, gpu> v) const {

  Vector3f<ad, gpu> pl = p - radius_;
  auto esc_length = ((p.y() < 0) & (v.y() < 0)) 
                  | ((p.y() > length_) & (v.y() > 0));

  auto out_radius = (pl.x() * pl.x() + pl.z() * pl.z()) >= (radius_ * radius_);
  auto esc_radius = (pl.x() * v.x() + pl.z() * v.z()) > 0;

  return (out_radius & esc_radius) | esc_length;
}

// Explicit Instantiations

// gpu
template struct cylinder_volume<true, true>;
template struct cylinder_volume<false, true>;

// cpu
template struct cylinder_volume<true, false>;
template struct cylinder_volume<false, false>;

} // namespace drrt

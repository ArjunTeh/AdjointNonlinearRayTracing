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

template <bool ad, bool gpu>
volume<ad, gpu>::volume() : res_(1, 1, 1), data_(0.0) {}

template <bool ad, bool gpu>
volume<ad, gpu>::volume(float value)
  : res_(1, 1, 1), data_(value) {}

template <bool ad, bool gpu>
volume<ad, gpu>::volume(int width, int height, int depth, const Float<ad, gpu> &data)
  : res_(width, height, depth), data_(data) {
  if (width * height * depth == static_cast<int>(slices(data))) {
    return;
  }
  throw std::runtime_error("Resolution doesn't match data");
}

template <bool ad, bool gpu>
volume<ad, gpu>::volume(ScalarVector3i res, const Float<ad, gpu> &data, scalar_t<Float<ad, gpu>> h)
    : h_(h), res_(res), data_(data) {
  if (res[0] * res[1] * res[2] == static_cast<int>(slices(data))) {
    return;
  }
  throw std::runtime_error("Resolution doesn't match data");
}

template <bool ad, bool gpu>
Matrix<Float<ad, gpu>, 3>
volume<ad, gpu>::eval_hess(Vector3f<ad, gpu> const& p,
                           mask_t<Float<ad, gpu>> const& mask) const {

  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;
  using myMatrix = Matrix<myFloat, 3>;

  const int width = res_.x();
  const int height = res_.y();
  const int depth = res_.z();

  fVector3 pm = p * rcp(h_);
  iVector3 pos = floor2int<iVector3, fVector3>(pm);
  fVector3 w0 = pm - fVector3(pos), w1 = 1.0f - w0;
  iVector3 pos0 = enoki::max(enoki::min(pos, res_ - 1), 0);
  iVector3 pos1 = enoki::max(enoki::min(pos+1, res_ - 1), 0);

  myInt idx000 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos0.x());
  myInt idx100 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos1.x());
  myInt idx010 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos0.x());
  myInt idx110 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos1.x());
  myInt idx001 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos0.x());
  myInt idx101 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos1.x());
  myInt idx011 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos0.x());
  myInt idx111 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos1.x());

  myFloat v000 = gather<myFloat>(data_, idx000, mask);
  myFloat v100 = gather<myFloat>(data_, idx100, mask);
  myFloat v010 = gather<myFloat>(data_, idx010, mask);
  myFloat v110 = gather<myFloat>(data_, idx110, mask);
  myFloat v001 = gather<myFloat>(data_, idx001, mask);
  myFloat v101 = gather<myFloat>(data_, idx101, mask);
  myFloat v011 = gather<myFloat>(data_, idx011, mask);
  myFloat v111 = gather<myFloat>(data_, idx111, mask);

  myFloat dxdy = lerp(v110 - v010 - v100 + v000, 
                      v111 - v011 - v101 + v001,
                      w0.z());
  myFloat dxdz = lerp(v101 - v001 - v100 + v000,
                      v111 - v011 - v110 + v010, 
                      w0.y());
  myFloat dydz = lerp(v011 - v001 - v010 + v000,
                      v111 - v101 - v110 + v100, 
                      w0.x());

  myMatrix H(0);
  set_slices(H, slices(v000));
  H(0, 1) = dxdy;
  H(0, 2) = dxdz;
  H(1, 0) = dxdy;
  H(1, 2) = dydz;
  H(2, 0) = dxdz;
  H(2, 1) = dydz;

  return H / h_ / h_;
}

template <bool ad, bool gpu>
std::pair<Float<ad, gpu>, Vector3f<ad, gpu>>
volume<ad, gpu>::eval_grad(Vector3f<ad, gpu> const& p, Mask const& mask) const {

  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;

  const int width = res_.x();
  const int height = res_.y();
  const int depth = res_.z();

  if (static_cast<int>(slices(data_)) != width * height * depth)
    throw std::runtime_error("volume: invalid data size!");

  if (width == 1 && height == 1 && depth == 1) {
    if constexpr (ad)
      return {data_, Vector3fD(0,0,0)};
    else
      return {detach(data_), Vector3fC(0, 0, 0)};
  } else {
    if (width < 2 || height < 2)
      throw std::runtime_error("volume: invalid resolution!");

    // fVector3 pm = p / h_ - Float<false, gpu>(0.5);
    //fVector3 pm = fmadd(p, rcp(h_), -Float<false, gpu>(0.5));
    fVector3 pm = p * rcp(h_);
    iVector3 pos = floor2int<iVector3, fVector3>(pm);
    fVector3 w0 = pm - fVector3(pos), w1 = 1.0f - w0;
    iVector3 pos0 = enoki::max(enoki::min(pos, res_ - 1), 0);
    iVector3 pos1 = enoki::max(enoki::min(pos+1, res_ - 1), 0);

    myInt idx000 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos0.x());
    myInt idx100 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos1.x());
    myInt idx010 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos0.x());
    myInt idx110 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos1.x());
    myInt idx001 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos0.x());
    myInt idx101 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos1.x());
    myInt idx011 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos0.x());
    myInt idx111 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos1.x());

    myFloat v000 = gather<myFloat>(data_, idx000, mask);
    myFloat v100 = gather<myFloat>(data_, idx100, mask);
    myFloat v010 = gather<myFloat>(data_, idx010, mask);
    myFloat v110 = gather<myFloat>(data_, idx110, mask);
    myFloat v001 = gather<myFloat>(data_, idx001, mask);
    myFloat v101 = gather<myFloat>(data_, idx101, mask);
    myFloat v011 = gather<myFloat>(data_, idx011, mask);
    myFloat v111 = gather<myFloat>(data_, idx111, mask);

    myFloat w000 = w1.x()*w1.y()*w1.z();
    myFloat w100 = w0.x()*w1.y()*w1.z();
    myFloat w010 = w1.x()*w0.y()*w1.z();
    myFloat w110 = w0.x()*w0.y()*w1.z();
    myFloat w001 = w1.x()*w1.y()*w0.z();
    myFloat w101 = w0.x()*w1.y()*w0.z();
    myFloat w011 = w1.x()*w0.y()*w0.z();
    myFloat w111 = w0.x()*w0.y()*w0.z();

    // Trilinear interpolation
    myFloat n = w000*v000 + w100*v100 + w010*v010 + w110*v110 +
                w001*v001 + w101*v101 + w011*v011 + w111*v111;

    myFloat nx =  (v100*w1.y()*w1.z() + v101*w1.y()*w0.z() +
                   v110*w0.y()*w1.z() + v111*w0.y()*w0.z())
                - (v000*w1.y()*w1.z() + v001*w1.y()*w0.z() +
                   v010*w0.y()*w1.z() + v011*w0.y()*w0.z());
    myFloat ny =  (v010*w1.x()*w1.z() + v011*w1.x()*w0.z() +
                   v110*w0.x()*w1.z() + v111*w0.x()*w0.z())
                - (v000*w1.x()*w1.z() + v001*w1.x()*w0.z() +
                   v100*w0.x()*w1.z() + v101*w0.x()*w0.z());
    myFloat nz =  (v001*w1.x()*w1.y() + v011*w1.x()*w0.y() +
                   v101*w0.x()*w1.y() + v111*w0.x()*w0.y())
                - (v000*w1.x()*w1.y() + v010*w1.x()*w0.y() +
                   v100*w0.x()*w1.y() + v110*w0.x()*w0.y());

    return std::make_pair(n, fVector3(nx, ny, nz) * rcp(h_));
  }

}
template <bool ad, bool gpu>
void volume<ad, gpu>::splat(Vector3f<ad, gpu> const& p,
                            Float<ad, gpu> const& val,
                            Vector3f<ad, gpu> const& grad,
                            mask_t<Float<ad, gpu>> active) {
  using myFloat = Float<ad, gpu>;
  using myInt = Int<ad, gpu>;
  using fVector3 = Vector3f<ad, gpu>;
  using iVector3 = Vector3i<ad, gpu>;

  const int width = res_.x();
  const int height = res_.y();
  const int depth = res_.z();

  if (static_cast<int>(slices(data_)) != width * height * depth)
    throw std::runtime_error("volume: invalid data size!");

  //fVector3 pm = p / h_ - Float<false, gpu>(0.5);
  fVector3 pm = p * rcp(h_);
  iVector3 pos = floor2int<iVector3, fVector3>(pm);
  fVector3 w0 = pm - fVector3(pos), w1 = 1.0f - w0;

  iVector3 pos0 = enoki::max(enoki::min(pos, res_ - 1), 0);
  iVector3 pos1 = enoki::max(enoki::min(pos+1, res_ - 1), 0);

  myInt idx000 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos0.x());
  myInt idx100 = fmadd(fmadd(pos0.z(), height, pos0.y()), width, pos1.x());
  myInt idx010 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos0.x());
  myInt idx110 = fmadd(fmadd(pos0.z(), height, pos1.y()), width, pos1.x());
  myInt idx001 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos0.x());
  myInt idx101 = fmadd(fmadd(pos1.z(), height, pos0.y()), width, pos1.x());
  myInt idx011 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos0.x());
  myInt idx111 = fmadd(fmadd(pos1.z(), height, pos1.y()), width, pos1.x());

  // splat val
  scatter_add(data_, val*w1.x()*w1.y()*w1.z(), idx000, active);
  scatter_add(data_, val*w0.x()*w1.y()*w1.z(), idx100, active);
  scatter_add(data_, val*w1.x()*w0.y()*w1.z(), idx010, active);
  scatter_add(data_, val*w0.x()*w0.y()*w1.z(), idx110, active);
  scatter_add(data_, val*w1.x()*w1.y()*w0.z(), idx001, active);
  scatter_add(data_, val*w0.x()*w1.y()*w0.z(), idx101, active);
  scatter_add(data_, val*w1.x()*w0.y()*w0.z(), idx011, active);
  scatter_add(data_, val*w0.x()*w0.y()*w0.z(), idx111, active);

  // splat grad
  myFloat v000 = -grad.x()*w1.y()*w1.z() - grad.y()*w1.x()*w1.z() - grad.z()*w1.x()*w1.y();
  myFloat v100 =  grad.x()*w1.y()*w1.z() - grad.y()*w0.x()*w1.z() - grad.z()*w0.x()*w1.y();
  myFloat v010 = -grad.x()*w0.y()*w1.z() + grad.y()*w1.x()*w1.z() - grad.z()*w1.x()*w0.y();
  myFloat v110 =  grad.x()*w0.y()*w1.z() + grad.y()*w0.x()*w1.z() - grad.z()*w0.x()*w0.y();
  myFloat v001 = -grad.x()*w1.y()*w0.z() - grad.y()*w1.x()*w0.z() + grad.z()*w1.x()*w1.y();
  myFloat v101 =  grad.x()*w1.y()*w0.z() - grad.y()*w0.x()*w0.z() + grad.z()*w0.x()*w1.y();
  myFloat v011 = -grad.x()*w0.y()*w0.z() + grad.y()*w1.x()*w0.z() + grad.z()*w1.x()*w0.y();
  myFloat v111 =  grad.x()*w0.y()*w0.z() + grad.y()*w0.x()*w0.z() + grad.z()*w0.x()*w0.y();

  scatter_add(data_, v000, idx000, active);
  scatter_add(data_, v100, idx100, active);
  scatter_add(data_, v010, idx010, active);
  scatter_add(data_, v110, idx110, active);
  scatter_add(data_, v001, idx001, active);
  scatter_add(data_, v101, idx101, active);
  scatter_add(data_, v011, idx011, active);
  scatter_add(data_, v111, idx111, active);
}

template <bool ad, bool gpu>
mask_t<Float<ad, gpu>> volume<ad, gpu>::inbounds(Vector3f<ad, gpu> p) const {
  // TODO(ateh): transform to local frame
  auto below = (p.x() >= 0) &
               (p.y() >= 0) &
               (p.z() >= 0);
  auto above = (p.x() < ((res_.x()-1)*h_)) &
               (p.y() < ((res_.y()-1)*h_)) &
               (p.z() < ((res_.z()-1)*h_));
  return below & above;
}

template <bool ad, bool gpu>
mask_t<Float<ad, gpu>> volume<ad, gpu>::escaped(Vector3f<ad, gpu> p,
    Vector3f<ad, gpu> v) const {
  
  // check the three axes
  auto x_esc = ((p.x() < 0) & (v.x() < 0)) 
             | ((p.x() >= ((res_.x()-1) * h_)) & (v.x() > 0));
  auto y_esc = ((p.y() < 0) & (v.y() < 0)) 
             | ((p.y() >= ((res_.y()-1) * h_)) & (v.y() > 0));
  auto z_esc = ((p.z() < 0) & (v.z() < 0)) 
             | ((p.z() >= ((res_.z()-1) * h_)) & (v.z() > 0));

  return x_esc | y_esc | z_esc;
}

// Explicit Instantiations

// gpu
template struct volume<true, true>;
template struct volume<false, true>;

// cpu
template struct volume<true, false>;
template struct volume<false, false>;


} // namespace drrt

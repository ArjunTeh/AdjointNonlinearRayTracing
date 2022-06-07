#pragma once

#include "eikonal.h"
#include <enoki/dynamic.h>
#include <enoki/cuda.h>

namespace drrt {

// TODO(ateh): Refactor the volumes to have an interface that the tracer can use
template <bool ad, bool gpu>
struct vol_interface {
  Float<ad, gpu> get_data() { return data_; }

  virtual
  std::pair<Float<ad, gpu>, Vector3f<ad, gpu>>
  eval_grad(Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const = 0;

  Float<ad, gpu> data_;
};

template <bool ad, bool gpu>
struct volume {
 public:

  using Mask = mask_t<Float<ad, gpu>>;

	volume();
	volume(float value);
	volume(int width, int height, int depth, const Float<ad, gpu> &data);
	volume(ScalarVector3i res, const Float<ad, gpu> &data, scalar_t<Float<ad, gpu>> h);

  Float<ad, gpu> get_data() { return data_; }

  std::pair<Float<ad, gpu>, Vector3f<ad, gpu>>
  eval_grad(Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const;

  Matrix<Float<ad, gpu>, 3>
  eval_hess(Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const;

  void splat(Vector3f<ad, gpu> const& pos, 
			 Float<ad, gpu> const& val, 
			 Vector3f<ad, gpu> const& grad,
			 Mask active = true);

  Mask inbounds(Vector3f<ad, gpu> p) const;
  Mask escaped(Vector3f<ad, gpu> p, Vector3f<ad, gpu> v) const;

  Float<false, gpu> h_;
  ScalarVector3i res_;
  Float<ad, gpu> data_;
};

template <bool ad, bool gpu>
struct cylinder_volume {
 public:
	using Mask = mask_t<Float<ad, gpu>>;

	cylinder_volume();
    cylinder_volume(const Float<ad, gpu> &data, 
					scalar_t<Float<ad, gpu>> radius, 
					scalar_t<Float<ad, gpu>> length);

    Float<ad, gpu> get_data() { return data_; }

	std::pair<Float<ad, gpu>, Vector3f<ad, gpu>>
	eval_grad(Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const;

	Matrix<Float<ad, gpu>, 3>
	eval_hess(Vector3f<ad, gpu> const& p, mask_t<Float<ad, gpu>> const& mask) const;

	void splat(Vector3f<ad, gpu> const& pos, 
	  		   Float<ad, gpu> const& val, 
	  		   Vector3f<ad, gpu> const& grad,
	  		   Mask active = true);

	Mask inbounds(Vector3f<ad, gpu> p) const;
	Mask escaped(Vector3f<ad, gpu> p, Vector3f<ad, gpu> v) const;

	Float<ad, gpu> data_;
	scalar_t<Float<ad, gpu>> radius_;
	scalar_t<Float<ad, gpu>> length_;
};


} // namespace drrt


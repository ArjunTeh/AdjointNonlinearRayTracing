#pragma once

#include "eikonal.h"
#include <iostream>
#include <enoki/dynamic.h>
#include <enoki/cuda.h>

namespace drrt {

using Floater = DynamicArray<Packet<float>>;
template <bool ad, bool gpu>
class Tracer {
 public:

  std::pair<Vector3f<ad, gpu>, Vector3f<ad, gpu>>
  trace(Float<ad, gpu>& rif,
        ScalarVector3i res,
        Vector3f<ad, gpu>& pos,
        Vector3f<ad, gpu>& vel,
        scalar_t<Float<ad, gpu>> h,
        scalar_t<Float<ad, gpu>> ds);

  std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Bool<ad, gpu>>
  trace_plane(Float<ad, gpu>& rif,
              ScalarVector3i res,
              Vector3f<ad, gpu>& pos,
              Vector3f<ad, gpu>& vel,
              Vector3f<ad, gpu>& pln_o,
              Vector3f<ad, gpu>& pln_d,
              scalar_t<Float<ad, gpu>> h,
              scalar_t<Float<ad, gpu>> ds);

  std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Float<ad, gpu>>
  trace_target(Float<ad, gpu>& rif,
               ScalarVector3i res,
               Vector3f<ad, gpu>& pos,
               Vector3f<ad, gpu>& vel,
               Vector3f<ad, gpu>& target,
               scalar_t<Float<ad, gpu>> h,
               scalar_t<Float<ad, gpu>> ds);

  std::pair<Vector3f<ad, gpu>, Vector3f<ad, gpu>>
  trace_sdf(Float<ad, gpu>& rif,
            Float<ad, gpu>& sdf,
            ScalarVector3i res,
            Vector3f<ad, gpu>& pos,
            Vector3f<ad, gpu>& vel,
            scalar_t<Float<ad, gpu>> h,
            scalar_t<Float<ad, gpu>> ds);

  std::tuple<Vector3f<ad, gpu>, Vector3f<ad, gpu>, Float<ad, gpu>>
  trace_cable(Float<ad, gpu>& rif,
              scalar_t<Float<ad, gpu>> radius,
              scalar_t<Float<ad, gpu>> length,
              Vector3f<ad, gpu>& pos,
              Vector3f<ad, gpu>& vel,
              Vector3f<ad, gpu>& target,
              scalar_t<Float<ad, gpu>> ds);

  Float<false, gpu>
  backtrace(Float<false, gpu>& rif,
            ScalarVector3i res,
            Vector3f<false, gpu>& xt,
            Vector3f<false, gpu>& vt,
            Vector3f<false, gpu>& dx,
            Vector3f<false, gpu>& dv,
            scalar_t<Float<false, gpu>> h,
            scalar_t<Float<false, gpu>> ds);
            
  Float<false, gpu>
  backtrace_sdf(Float<false, gpu>& rif,
                Float<false, gpu>& sdf,
                ScalarVector3i res,
                Vector3f<false, gpu>& xt,
                Vector3f<false, gpu>& vt,
                Vector3f<false, gpu>& dx,
                Vector3f<false, gpu>& dv,
                scalar_t<Float<false, gpu>> h,
                scalar_t<Float<false, gpu>> ds);

  Float<false, gpu>
  backtrace_cable(Float<false, gpu>& rif,
                  scalar_t<Float<false, gpu>> radius,
                  scalar_t<Float<false, gpu>> length,
                  Vector3f<false, gpu>& xt,
                  Vector3f<false, gpu>& vt,
                  Vector3f<false, gpu>& dx,
                  Vector3f<false, gpu>& dv,
                  scalar_t<Float<ad, gpu>> ds);

  void test_in(Vector3f<ad, gpu> p);
  Vector3fC tester();
};


} // namespace drrt

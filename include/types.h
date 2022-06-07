#pragma once

// #include <enoki/matrix.h>
#include <enoki/cuda.h>
#include <enoki/autodiff.h>
#include <enoki/dynamic.h>
#include <enoki/array.h>

using namespace enoki;

namespace drrt {

/********************************************
 * GPU array types
 ********************************************/

template <typename T, bool ad>
using GPUType      = typename std::conditional<ad,
                                            DiffArray<CUDAArray<T>>,
                                            CUDAArray<T>>::type;

template <typename T, bool ad>
using CPUType      = typename std::conditional<ad,
                                               DiffArray<DynamicArray<Packet<T>>>,
                                               DynamicArray<Packet<T>>>::type;

template <typename T, bool ad, bool gpu>
using Type         = typename std::conditional<gpu,
                                               GPUType<T, ad>,
                                               CPUType<T, ad>>::type;



// Scalar arrays (GPU)

template <bool ad, bool gpu>
using Float     = Type<float, ad, gpu>;

template <bool ad, bool gpu>
using Bool      = Type<bool, ad, gpu>;

//template <bool ad, bool gpu>
//using Double     = Type<double, ad, gpu>;

template <bool ad, bool gpu>
using Int       = Type<int32_t, ad, gpu>;

using BoolC     = Bool<false, true>;

using FloatC    = Float<false, true>;
using FloatD    = Float<true, true>;

template <bool ad>
using FloatS    = Float<ad, false>;

using IntC      = Int<false, true>;
using IntD      = Int<true, true>;

template <bool ad>
using IntS      = Int<ad, false>;

// Vector arrays

template <int n, bool ad, bool gpu>
using Vectorf   = Array<Float<ad, gpu>, n>;

template <int n, bool ad, bool gpu>
using Vectori   = Array<Int<ad, gpu>, n>;

// template <int n, bool ad>
// using Matrixf   = Matrix<Float<ad>, n>;

template <bool ad, bool gpu>
using Vector2f  = Vectorf<2, ad, gpu>;

template <bool ad, bool gpu>
using Vector2i  = Vectori<2, ad, gpu>;

template <bool ad, bool gpu>
using Vector3f  = Vectorf<3, ad, gpu>;

template <bool ad, bool gpu>
using Vector3i  = Vectori<3, ad, gpu>;

// GPU Vectors
using Vector2fC = Vector2f<false, true>;
using Vector2fD = Vector2f<true, true>;

using Vector2iC = Vector2i<false, true>;
using Vector2iD = Vector2i<true, true>;

using Vector3fC = Vector3f<false, true>;
using Vector3fD = Vector3f<true, true>;

using Vector3iC = Vector3i<false, true>;
using Vector3iD = Vector3i<true, true>;

using Vector4iC = Vectori<4, false, true>;

using Vector4fC = Vectorf<4, false, true>;
using Vector4fD = Vectorf<4, true, true>;

// CPU Vectors
template <bool ad>
using SVector2f = Vector2f<ad, false>;

template <bool ad>
using SVector2i = Vector2i<ad, false>;

template <bool ad>
using SVector3f = Vector3f<ad, false>;

template <bool ad>
using SVector3i = Vector3i<ad, false>;

// Matrix arrays (GPU)

// template <bool ad>
// using Matrix3f  = Matrixf<3, ad>;

// template <bool ad>
// using Matrix4f  = Matrixf<4, ad>;

// using Matrix3fC = Matrix3f<false>;
// using Matrix3fD = Matrix3f<true>;

// using Matrix4fC = Matrix4f<false>;
// using Matrix4fD = Matrix4f<true>;

/********************************************
 * CPU types
 ********************************************/

// Static Types
using ScalarVector2f = Array<float, 2>;
using ScalarVector3f = Array<float, 3>;
using ScalarVector4f = Array<float, 4>;

using ScalarVector2i = Array<int, 2>;
using ScalarVector3i = Array<int, 3>;
using ScalarVector4i = Array<int, 4>;

// using ScalarMatrix2f = Matrix<float, 2>;
// using ScalarMatrix3f = Matrix<float, 3>;
// using ScalarMatrix4f = Matrix<float, 4>;

} // namespace drt

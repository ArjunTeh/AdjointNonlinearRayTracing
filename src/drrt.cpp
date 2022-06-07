#include <eikonal.h>
#include <tracer.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <enoki/stl.h>
#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/python.h>

#include <iostream>

/* Import pybind11 and Enoki namespaces */

namespace py = pybind11;
using namespace py::literals;

using namespace drrt;

PYBIND11_MODULE(drrt, m) {
  py::module::import("enoki");
  py::module::import("enoki.cuda");
  py::module::import("enoki.cuda_autodiff");

  m.doc() = "Differentiable Refractive Ray Tracing"; // Set a docstring

  py::class_<drrt::Tracer<true, true>>(m, "TracerD")
      .def(py::init<>())
      .def("test", &Tracer<true, true>::tester)
      .def("testscale", &Tracer<true, true>::test_in)
      .def("trace", &Tracer<true, true>::trace)
      .def("trace_pln", &Tracer<true, true>::trace_plane)
      .def("trace_target", &Tracer<true, true>::trace_target)
      .def("trace_sdf", &Tracer<true, true>::trace_sdf)
      .def("trace_cable", &Tracer<true, true>::trace_cable);

  py::class_<drrt::Tracer<false, false>>(m, "TracerS")
      .def(py::init<>())
      .def("test", &Tracer<false, false>::tester)
      .def("testscale", &Tracer<false, false>::test_in)
      .def("trace", &Tracer<false, false>::trace)
      .def("trace_sdf", &Tracer<false, false>::trace_sdf)
      .def("trace_target", &Tracer<false, false>::trace_target)
      .def("backtrace", &Tracer<false, false>::backtrace);

  py::class_<drrt::Tracer<false, true>>(m, "TracerC")
      .def(py::init<>())
      .def("test", &Tracer<false, true>::tester)
      .def("testscale", &Tracer<false, true>::test_in)
      .def("trace", &Tracer<false, true>::trace)
      .def("trace_pln", &Tracer<false, true>::trace_plane)
      .def("trace_sdf", &Tracer<false, true>::trace_sdf)
      .def("trace_target", &Tracer<false, true>::trace_target)
      .def("trace_cable", &Tracer<false, true>::trace_cable)
      .def("backtrace", &Tracer<false, true>::backtrace)
      .def("backtrace_sdf", &Tracer<false, true>::backtrace_sdf)
      .def("backtrace_cable", &Tracer<false, true>::backtrace_cable);

}

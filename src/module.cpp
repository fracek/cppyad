/* Copyright 2020 Francesco Ceccon
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================
 */
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <cppad/cppad.hpp>
#include <cppad/utility/to_string.hpp>

#include "numpy.hpp"
#include "func.hpp"


namespace py = pybind11;


template<class U>
std::vector<CppAD::AD<U>> independent(const std::vector<U>& xs) {
  std::vector<CppAD::AD<U>> ax(xs.size());
  for (auto i = 0UL; i < xs.size(); ++i) {
    ax[i] = xs[i];
  }
  CppAD::Independent(ax);
  return ax;
}


template<class U>
U value(const CppAD::AD<U>& x) {
  return CppAD::Value(x);
}


template<class U>
bool is_variable(const CppAD::AD<U>& x) {
  return CppAD::Variable(x);
}


template<class U>
void init_module(py::module m) {
  py::class_<CppAD::AD<U>>(m, "AD")
    .def(+py::self)
    .def(-py::self)
    .def(py::self *= py::self)
    .def(py::self += py::self)
    .def(py::self /= py::self)
    .def(py::self -= py::self)
    .def(py::self *= double())
    .def(py::self += double())
    .def(py::self /= double())
    .def(py::self -= double())
    .def(py::self * py::self)
    .def(py::self + py::self)
    .def(py::self / py::self)
    .def(py::self - py::self)
    .def(double() * py::self)
    .def(double() + py::self)
    .def(double() / py::self)
    .def(double() - py::self)
    .def(py::self * double())
    .def(py::self + double())
    .def(py::self / double())
    .def(py::self - double())
    .def(py::self > py::self)
    .def(py::self >= py::self)
    .def(py::self != py::self)
    .def("__pow__",
	 [](const CppAD::AD<U>& v, U e) {
	   return CppAD::pow(v, e);
	 })
    .def("__str__",
	 [](const CppAD::AD<U>& v) {
	   return CppAD::to_string(v);
	 });

  py::class_<ADFunc<U>>(m, "ADFun")
    .def(py::init<const std::vector<CppAD::AD<U>>&, const std::vector<CppAD::AD<U>> &>())
    .def("forward", &ADFunc<U>::forward)
    .def("reverse", &ADFunc<U>::reverse)
    .def("jacobian", &ADFunc<U>::jacobian)
    .def("sparse_jacobian_reverse", &ADFunc<U>::sparse_jacobian_reverse)
    .def("sparse_jacobian_forward", &ADFunc<U>::sparse_jacobian_forward)
    .def("sparse_jacobian_pattern_reverse", &ADFunc<U>::sparse_jacobian_pattern_reverse)
    .def("sparse_jacobian_pattern_forward", &ADFunc<U>::sparse_jacobian_pattern_forward)
    .def("sparse_hessian_pattern_reverse", &ADFunc<U>::sparse_hessian_pattern_reverse)
    .def("sparse_hessian", &ADFunc<U>::sparse_hessian)
    .def("hessian", py::overload_cast<numpy_array<U> *, std::size_t>(&ADFunc<U>::hessian))
    .def("hessian", py::overload_cast<numpy_array<U> *, numpy_array<U> *>(&ADFunc<U>::hessian))
    .def_property_readonly("range", &ADFunc<U>::range)
    .def_property_readonly("domain", &ADFunc<U>::domain);

  m.def("independent", &independent<U>);
  m.def("value", &value<U>);
  m.def("is_variable", &is_variable<U>);

}


PYBIND11_MODULE(cppyad_core, m) {
  init_module<double>(m);

  py::class_<CppAD::sparse_jacobian_work>(m, "SparseJacobianWork")
    .def(py::init());
  py::class_<CppAD::sparse_hessian_work>(m, "SparseHessianWork")
    .def(py::init());
}

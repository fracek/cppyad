#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <cppad/utility/to_string.hpp>

#include "environment.hpp"
#include "func.hpp"

namespace py = pybind11;


AD_double_vector independent(const std::vector<double>& xs) {
    std::vector<AD_double> ax(xs.size());
    for (auto i = 0UL; i < xs.size(); ++i) {
        ax[i] = xs[i];
    }
    CppAD::Independent(ax);
    return ax;
}

double value(const AD_double& x) {
    return CppAD::Value(x);
}

bool is_variable(const AD_double& x) {
    return CppAD::Variable(x);
}


#define FORWARD_UNARY_FUNCTION(m, name) \
    m.def(#name, (AD_double (*)(const AD_double &)) &CppAD::name)

#define FORWARD_BINARY_FUNCTION(m, name) \
    m.def(#name, (AD_double (*)(const AD_double &, const AD_double &)) &CppAD::name); \
    m.def(#name, (AD_double (*)(const AD_double &, const double &)) &CppAD::name); \
    m.def(#name, (AD_double (*)(const double &, const AD_double &)) &CppAD::name);

template<class U>
void init_adfunc(py::module &m, const char *name) {
  py::class_<ADFunc<U>>(m, name)
    .def(py::init<const std::vector<CppAD::AD<U>>&, const std::vector<CppAD::AD<U>> &>())
    .def("forward", &ADFunc<U>::forward)
    .def("reverse", &ADFunc<U>::reverse)
    .def("jacobian", &ADFunc<U>::jacobian)
    .def("sparse_jacobian_reverse", &ADFunc<U>::sparse_jacobian_reverse)
    .def("sparse_jacobian_forward", &ADFunc<U>::sparse_jacobian_forward)
    .def("sparse_jacobian_pattern", &ADFunc<U>::sparse_jacobian_pattern)
    .def("hessian", py::overload_cast<const std::vector<U>&, std::size_t>(&ADFunc<U>::hessian))
    .def("hessian", py::overload_cast<const std::vector<U>&, const std::vector<U>&>(&ADFunc<U>::hessian))
    .def_property_readonly("range", &ADFunc<U>::range)
    .def_property_readonly("domain", &ADFunc<U>::domain);

  py::class_<VectorArrayWrapper<U>>(m, "VectorArrayWrapper")
    .def(py::init<std::size_t>())
    .def("as_pyarray", &VectorArrayWrapper<U>::as_pyarray);
}


PYBIND11_MODULE(cppyad_core, m) {
    py::class_<AD_double>(m, "AD_double")
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
        .def("__str__", [](const AD_double& v) {
            return CppAD::to_string(v);
        });

    init_adfunc<double>(m, "ADFun_double");

    m.def("independent", &independent);
    m.def("value", &value);
    m.def("is_variable", &is_variable);
    FORWARD_UNARY_FUNCTION(m, abs);
    FORWARD_UNARY_FUNCTION(m, acos);
    FORWARD_UNARY_FUNCTION(m, acosh);
    FORWARD_UNARY_FUNCTION(m, asin);
    FORWARD_UNARY_FUNCTION(m, asinh);
    FORWARD_UNARY_FUNCTION(m, atan);
    //FORWARD_UNARY_FUNCTION(m, atan2);
    FORWARD_UNARY_FUNCTION(m, atanh);
    //FORWARD_UNARY_FUNCTION(m, conj);
    FORWARD_UNARY_FUNCTION(m, cos);
    FORWARD_UNARY_FUNCTION(m, cosh);
    FORWARD_UNARY_FUNCTION(m, erf);
    FORWARD_UNARY_FUNCTION(m, exp);
    FORWARD_UNARY_FUNCTION(m, expm1);
    FORWARD_UNARY_FUNCTION(m, fabs);
    FORWARD_UNARY_FUNCTION(m, log);
    FORWARD_UNARY_FUNCTION(m, log10);
    FORWARD_UNARY_FUNCTION(m, log1p);
    FORWARD_UNARY_FUNCTION(m, sign);
    FORWARD_UNARY_FUNCTION(m, sin);
    FORWARD_UNARY_FUNCTION(m, sinh);
    FORWARD_UNARY_FUNCTION(m, sqrt);
    FORWARD_UNARY_FUNCTION(m, tan);
    FORWARD_UNARY_FUNCTION(m, tanh);
    FORWARD_BINARY_FUNCTION(m, pow);

    py::class_<CppAD::sparse_jacobian_work>(m, "SparseJacobianWork")
        .def(py::init());
}

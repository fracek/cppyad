#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*
 * This class is needed to avoid unnecessary allocations when computing e.g. sparse jacobians.
 * CppAD expects to take as input a mutable reference to a std::vector-like class, this behaviour does not play well
 * with Python. If we call the function with a list/numpy array the object will be copied and then passed to CppAD,
 * so we won't be able to access the result. If we try to pass a numpy array directly to CppAD, CppAD will complain
 * because it expects a vector-like object.
 *
 * The solution is to store data in a vector, and wrap this vector data in a numpy array so it can be used by Python.
 */
template<class U>
class VectorArrayWrapper {
public:
    VectorArrayWrapper(std::size_t n) : vec(nullptr) {
        vec = new std::vector<U>(n);
    }

    py::array_t<U> as_pyarray() {
        auto capsule = py::capsule(vec, [](void *v) { /*delete reinterpret_cast<std::vector<U>*>(v);*/ });
        return py::array(vec->size(), vec->data(), capsule);
    }

    std::vector<U> *vec;
};


template<class U>
class ADFunc {
public:
  ADFunc(const std::vector<CppAD::AD<U>>& x, const std::vector<CppAD::AD<U>>& y) {
    f_ = std::make_unique<CppAD::ADFun<U>>(x, y);
  }

  ADFunc(ADFunc&& other) : f_(std::move(other.f_)) {}
  ADFunc(ADFunc& other) = delete;

  std::vector<U> forward(std::size_t q, const std::vector<U>& x) {
    return f_->Forward(q, x);
  }

  std::vector<U> reverse(std::size_t q, const std::vector<U>& w) {
    return f_->Reverse(q, w);
  }

  std::vector<U> jacobian(const std::vector<U>& x) {
    return f_->Jacobian(x);
  }

  std::size_t sparse_jacobian_reverse(const std::vector<U>& x, const std::vector<bool>& p,
                                      const std::vector<std::size_t>& row, const std::vector<std::size_t>& col,
                                      VectorArrayWrapper<U> *jac_w, CppAD::sparse_jacobian_work *work) {
    return f_->SparseJacobianReverse(x, p, row, col, *jac_w->vec, *work);
  }

  std::size_t sparse_jacobian_forward(const std::vector<U>& x, const std::vector<bool>& p,
                                      const std::vector<std::size_t>& row, const std::vector<std::size_t>& col,
                                      VectorArrayWrapper<U> *jac_w, CppAD::sparse_jacobian_work *work) {
    return f_->SparseJacobianForward(x, p, row, col, *jac_w->vec, *work);
  }

  std::vector<bool> sparse_jacobian_pattern(std::size_t q, const std::vector<bool>& r) {
  return f_->RevSparseJac(q, r);
  }


  std::vector<U> hessian(const std::vector<U>& x, std::size_t l) {
    return f_->Hessian(x, l);
  }

  std::vector<U> hessian(const std::vector<U>& x, const std::vector<U>& w) {
    return f_->Hessian(x, w);
  }

  std::size_t range() const {
    return f_->Range();
  }

  std::size_t domain() const {
    return f_->Domain();
  }

private:
  std::unique_ptr<CppAD::ADFun<U>> f_;
};
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

#pragma once

#include <cppad/cppad.hpp>

#include "numpy.hpp"


/*
 * ADFunc is responsible for preparing ArrayVectorAdapter arguments and forward
 * them to CppAD.
 */
template<class U>
class ADFunc {
public:
  ADFunc(const std::vector<CppAD::AD<U>>& x, const std::vector<CppAD::AD<U>>& y) {
    f_ = std::make_unique<CppAD::ADFun<U>>(x, y);
  }

  ADFunc(ADFunc&& other) : f_(std::move(other.f_)) {}
  ADFunc(ADFunc& other) = delete;

  numpy_array<U> forward(std::size_t q, numpy_array<U> *x) {
    auto x_ = ArrayVectorAdapter<U>(x->mutable_data(), x->size());
    auto r =  f_->Forward(q, x_);
    return r.array();
  }

  numpy_array<U> reverse(std::size_t q, numpy_array<U> *w) {
    auto w_ = ArrayVectorAdapter<U>(w->mutable_data(), w->size());
    auto r = f_->Reverse(q, w_);
    return r.array();
  }

  numpy_array<U> jacobian(numpy_array<U> *x) {
    auto x_ = ArrayVectorAdapter<U>(x->mutable_data(), x->size());
    auto r = f_->Jacobian(x_);
    return r.array();
  }

  std::size_t sparse_jacobian_reverse(numpy_array<U> *x, numpy_array<bool> *p,
                                      numpy_array<std::size_t> *row, numpy_array<std::size_t> *col,
                                      numpy_array<U> *jac, CppAD::sparse_jacobian_work *work) {
    ArrayVectorAdapter<U> x_(x->mutable_data(), x->size());
    ArrayVectorAdapter<bool> p_(p->mutable_data(), p->size());
    ArrayVectorAdapter<std::size_t> row_(row->mutable_data(), row->size());
    ArrayVectorAdapter<std::size_t> col_(col->mutable_data(), col->size());
    ArrayVectorAdapter<U> jac_(jac->mutable_data(), jac->size());
    return f_->SparseJacobianReverse(x_, p_, row_, col_, jac_, *work);
  }

  std::size_t sparse_jacobian_forward(numpy_array<U> *x, numpy_array<bool> *p,
                                      numpy_array<std::size_t> *row, numpy_array<std::size_t> *col,
                                      numpy_array<U> *jac, CppAD::sparse_jacobian_work *work) {
    ArrayVectorAdapter<U> x_(x->mutable_data(), x->size());
    ArrayVectorAdapter<bool> p_(p->mutable_data(), p->size());
    ArrayVectorAdapter<std::size_t> row_(row->mutable_data(), row->size());
    ArrayVectorAdapter<std::size_t> col_(col->mutable_data(), col->size());
    ArrayVectorAdapter<U> jac_(jac->mutable_data(), jac->size());
    return f_->SparseJacobianForward(x_, p_, row_, col_, jac_, *work);
  }

  numpy_array<bool> sparse_jacobian_pattern_reverse(std::size_t q, numpy_array<bool> *r) {
    ArrayVectorAdapter<bool> r_(r->mutable_data(), r->size());
    auto res = f_->RevSparseJac(q, r_);
    return res.array();
  }

  numpy_array<bool> sparse_jacobian_pattern_forward(std::size_t q, numpy_array<bool> *r) {
    ArrayVectorAdapter<bool> r_(r->mutable_data(), r->size());
    auto res = f_->ForSparseJac(q, r_);
    return res.array();
  }

  numpy_array<bool> sparse_hessian_pattern_reverse(std::size_t q, numpy_array<bool> *s, bool transpose) {
    ArrayVectorAdapter<bool> s_(s->mutable_data(), s->size());
    auto res = f_->RevSparseHes(q, s_, transpose);
    return res.array();
  }

  std::size_t sparse_hessian(numpy_array<U> *x, numpy_array<U> *w, numpy_array<bool> *p,
                             numpy_array<std::size_t> *row, numpy_array<std::size_t> *col,
                             numpy_array<U> *hes, CppAD::sparse_hessian_work *work) {
    ArrayVectorAdapter<U> x_(x->mutable_data(), x->size());
    ArrayVectorAdapter<U> w_(w->mutable_data(), w->size());
    ArrayVectorAdapter<bool> p_(p->mutable_data(), p->size());
    ArrayVectorAdapter<std::size_t> row_(row->mutable_data(), row->size());
    ArrayVectorAdapter<std::size_t> col_(col->mutable_data(), col->size());
    ArrayVectorAdapter<U> hes_(hes->mutable_data(), hes->size());
    return f_->SparseHessian(x_, w_, p_, row_, col_, hes_, *work);
  }

  numpy_array<U> hessian(numpy_array<U> *x, std::size_t l) {
    ArrayVectorAdapter<U> x_(x->mutable_data(), x->size());
    auto res = f_->Hessian(x_, l);
    return res.array();
  }

  numpy_array<U> hessian(numpy_array<U> *x, numpy_array<U> *w) {
    ArrayVectorAdapter<U> x_(x->mutable_data(), x->size());
    ArrayVectorAdapter<U> w_(w->mutable_data(), w->size());
    auto res = f_->Hessian(x_, w_);
    return res.array();
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

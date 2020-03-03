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

#include <pybind11/numpy.h>

namespace py = pybind11;


template<class U>
using numpy_array = py::array_t<U, py::array::c_style>;


#pragma GCC visibility push(hidden)
template<class U>
class ArrayVectorAdapter {
public:
  using value_type = U;

  ArrayVectorAdapter() : arr_() {}
  ArrayVectorAdapter(std::size_t size) : arr_(size) {}

  // Why py::str?
  // See: https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
  // TLDR: someone has to own the data so it won't be copied.
  ArrayVectorAdapter(U *ptr, std::size_t size) : arr_(size, ptr, py::str()) {}

  std::size_t size() const { return arr_.size(); };

  U& operator[](std::size_t i) {
    return arr_.mutable_data()[i];
  }

  const U& operator[](std::size_t i) const {
    return arr_.data()[i];
  }

  void resize(std::size_t count) {
    std::vector<std::size_t> shape = { count };
    arr_.resize(shape);
  }

  numpy_array<U> array() {
    return std::move(arr_);
  }
private:
  numpy_array<U> arr_;
};
#pragma GCC visibility pop

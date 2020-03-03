#pragma once

#include <vector>

#include <cppad/cppad.hpp>


typedef std::vector<double> double_vector;
typedef CppAD::AD<double> AD_double;
typedef std::vector<CppAD::AD<double>> AD_double_vector;
typedef CppAD::ADFun<double> ADFun_double;
//typedef CppAD::AD<AD_double> AD_AD_double;

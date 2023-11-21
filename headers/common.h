#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <assert.h>
#include <eigen3/Eigen/Dense>
#include "im2col.h"
#include "col2im.h"
#include "../datatypes/fixed.hpp"
using namespace std;
using namespace chrono;
using namespace Eigen;

namespace simple_nn
{
    template<typename T>
	using MatX = Matrix<T, Dynamic, Dynamic, RowMajor>;
    template<typename T>
    using VecX = Matrix<T, Dynamic, 1>;
    template<typename T>
    using RowVecX = Matrix<T, 1, Dynamic>;
    using MatXf = MatX<float>;
	typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatXi;
	typedef Matrix<int, Dynamic, 1> VecXi;
    /* using F = Share<float>; */
    /* using S32 = Share<uint32_t>; */
    /* using S64 = Share<uint64_t>; */
    /* using MART = Wrapper<float,int64_t,uint64_t,ANOTHER_FRACTIONAL_VALUE,uint64_t>; */
    /* using ART = Wrapper<float,int64_t,uint64_t,ANOTHER_FRACTIONAL_VALUE,uint64_t>; */
    /* using FCART = Wrapper<float,float,float,ANOTHER_FRACTIONAL_VALUE,float>; */
    /* using FCART = Wrapper<float,int64_t,uint64_t,ANOTHER_FRACTIONAL_VALUE,uint64_t>; */

    template<typename T>
	void write_file(const MatX<T>& data, int channels, string fname)
	{
		ofstream fout(fname, std::ios::app);
		if (channels != 0) {
			int n_row = (int)data.rows() / channels; // batch size
			int n_col = (int)data.cols() * channels; // feature size * channels
			for (int i = 0; i < n_row; i++) {
				const float* begin = data.data() + n_col * i;
				for (int j = 0; j < n_col; j++) {
					fout << *begin;
					if (j == n_col - 1) {
						fout << endl;
					}
					else {
						fout << ',';
					}
					begin++;
				}
			}
		}
		else {
			for (int i = 0; i < data.rows(); i++) {
				for (int j = 0; j < data.cols(); j++) {
					fout << data(i, j);
					if (j == data.cols() - 1) {
						fout << endl;
					}
					else {
						fout << ',';
					}
				}
			}
		}
		fout.close();
	}

    template<typename T>
	void init_weight(MatX<T>& W_OG, int fan_in, int fan_out, string option)
	{
        MatX<float> W(W_OG.rows(), W_OG.cols());
		unsigned seed = 42;
		default_random_engine e(seed);

		if (option == "lecun_normal") {
			float s = std::sqrt(1.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "lecun_uniform") {
			float r = std::sqrt(1.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "xavier_normal") {
			float s = std::sqrt(2.f / (fan_in + fan_out));
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "xavier_uniform") {
			float r = std::sqrt(6.f / (fan_in + fan_out));
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "kaiming_normal") {
			float s = std::sqrt(2.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "kaiming_uniform") {
			float r = std::sqrt(6.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "normal") {
			normal_distribution<float> dist(0.f, 0.1f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "uniform") {
			uniform_real_distribution<float> dist(-0.01f, 0.01f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else {
			cout << "Invalid initialization." << endl;
			exit(1);
		}

    for (int i = 0; i < W.rows(); ++i) {
        for (int j = 0; j < W.cols(); ++j) {
            W_OG(i, j) = T(W(i, j));
        }
    }
}

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}
}

#pragma once
#include "layer.h"

namespace simple_nn
{
	const float FLOAT_MIN = -1000000000.f;

    template<typename T>
	class MaxPool2d : public Layer<T>
	{
	private:
		int batch;
		int ch;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int stride;
		MatX<T> im_col;
		vector<int> indices;
	public:
		MaxPool2d(int kernel_size, int stride);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	MaxPool2d<T>::MaxPool2d(int kernel_size, int stride) :
		Layer<T>(LayerType::MAXPOOL2D),
		batch(0),
		ch(0),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		stride(stride) {}

    template<typename T>
	void MaxPool2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ch = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, stride, 0);
		ow = calc_outsize(iw, kw, stride, 0);
		ohw = oh * ow;

		this->output.resize(batch * ch, ohw);
		this->delta.resize(batch * ch, ohw);
		im_col.resize(kh * kw, ohw);
		indices.resize(batch * ch * ohw);
	}

    template<typename T>
	void MaxPool2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		/* T* out = this->output.data(); */
		/* const T* pout = prev_out.data(); */
		/* for (int n = 0; n < batch; n++) { */
		/* 	for (int c = 0; c < ch; c++) { */
		/* 		for (int i = 0; i < oh; i++) { */
		/* 			for (int j = 0; j < ow; j++) { */
		/* 				int out_idx = j + ow * (i + oh * (c + ch * n)); */
		/* 				T max = T(FLOAT_MIN); */
		/* 				int max_idx = -1; */
		/* 				for (int y = 0; y < kh; y++) { */
		/* 					for (int x = 0; x < kw; x++) { */
		/* 						int ii = i * stride + y; */
		/* 						int jj = j * stride + x; */
		/* 						int pout_idx = jj + iw * (ii + ih * (c + ch * n)); */
		/* 						T val = T(FLOAT_MIN); */
		/* 						if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) { */
		/* 							val = pout[pout_idx]; */
		/* 						} */
		/* 						if (val > max) { */
		/* 							max = val; */
		/* 							max_idx = pout_idx; */
		/* 						} */
		/* 					} */
		/* 				} */
		/* 				out[out_idx] = max; */
		/* 				indices[out_idx] = max_idx; */
		/* 			} */
		/* 		} */
		/* 	} */
		/* } */
	}

    template<typename T>
	void MaxPool2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		T* pd = prev_delta.data();
		const T* d = this->delta.data();
		for (int i = 0; i < indices.size(); i++) {
			pd[indices[i]] += d[i];
		}
	}

    template<typename T>
	void MaxPool2d<T>::zero_grad() { this->delta.setZero(); }

    template<typename T>
	vector<int> MaxPool2d<T>::output_shape() { return { batch, ch, oh, ow }; }
}

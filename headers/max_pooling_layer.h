#pragma once
#include "layer.h"

namespace simple_nn
{
	const float FLOAT_MIN = -1000000000.f;

    template<typename T, int m = REDUCED_BITLENGTH_m, int k = REDUCED_BITLENGTH_k>
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
        int pad;
		MatX<T> im_col;
		vector<int> indices;
	public:
		MaxPool2d(int kernel_size, int stride, int pad = 0);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T, int m, int k>
	MaxPool2d<T,m,k>::MaxPool2d(int kernel_size, int stride, int pad) :
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
		stride(stride),
        pad(pad){}

    template<typename T, int m, int k>
	void MaxPool2d<T,m,k>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ch = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, stride, pad);
		ow = calc_outsize(iw, kw, stride, pad);
		ohw = oh * ow;

		this->output.resize(batch * ch, ohw);
		this->delta.resize(batch * ch, ohw);
		im_col.resize(kh * kw, ohw);
		indices.resize(batch * ch * ohw);
	}

    template<typename T, int m, int k>
	void MaxPool2d<T,m,k>::forward(const MatX<T>& prev_out, bool is_training)
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
        
		T* out = this->output.data();
		const T* pout = prev_out.data();
        auto max_candidates = new T[batch * ch * oh * ow * kh * kw];
        int counter = 0;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				for (int i = 0; i < oh; i++) {
					for (int j = 0; j < ow; j++) {
						/* int out_idx = j + ow * (i + oh * (c + ch * n)); */
						/* T max = T(0)-T(1); // T(FLOAT_MIN); */
						/* int max_idx = -1; */
						for (int y = 0; y < kh; y++) {
							for (int x = 0; x < kw; x++) {
								int ii = i * stride + y - pad;
								int jj = j * stride + x - pad;
								int pout_idx = jj + iw * (ii + ih * (c + ch * n));
								T val = T(UINT_TYPE(1) << (BITLENGTH - 1 )); // T(FLOAT_MIN);
								/* T val = T(FLOAT_MIN); */
								if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
									val = pout[pout_idx];
                                else if(pad > 0)
                                    val = T(0); // 0 padding
                                max_candidates[counter++] = val;
								/* if (val > max) { */
								/* 	max = val; */
								/* 	max_idx = pout_idx; */
								/* } */
							}
						}
						/* out[out_idx] = max; */
						/* indices[out_idx] = max_idx; */
					}
				}
			}
		}
    /* max_min_sint<0,BITLENGTH>(max_candidates, kh * kw, out, batch * ch * oh * ow, true); */
    max_min_sint<m,k>(max_candidates, kh * kw, out, batch * ch * oh * ow, true);
    delete[] max_candidates;
	}

    template<typename T, int m, int k>
	void MaxPool2d<T,m,k>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		T* pd = prev_delta.data();
		const T* d = this->delta.data();
		for (int i = 0; i < indices.size(); i++) {
			pd[indices[i]] += d[i];
		}
	}

    template<typename T, int m, int k>
	void MaxPool2d<T,m,k>::zero_grad() { this->delta.setZero(); }

    template<typename T, int m , int k>
	vector<int> MaxPool2d<T,m,k>::output_shape() { return { batch, ch, oh, ow }; }
}

#pragma once
#include "layer.h"
#include <chrono>

namespace simple_nn
{
    template<typename T>
	class AvgPool2d : public Layer<T>
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
		// MatX<T> im_col;
	public:
		AvgPool2d(int kernel_size, int stride);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	AvgPool2d<T>::AvgPool2d(int kernel_size, int stride) :
		Layer<T>(LayerType::AVGPOOL2D),
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
	void AvgPool2d<T>::set_layer(const vector<int>& input_shape)
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
		// im_col.resize(kh * kw, ohw);
	}

    template<typename T>
	void AvgPool2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    /* if(current_phase == 1) */
    /* std::cout << "AVG Pool ..." << std::endl; */
        T::communicate();
        this->output.setZero();
		T* out = this->output.data();
		const T* pout = prev_out.data();
		float denominator = kh * kw;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				for (int i = 0; i < oh; i++) {
					for (int j = 0; j < ow; j++) {
						int out_idx = j + ow * (i + oh * (c + ch * n));
						for (int y = 0; y < kh; y++) {
							for (int x = 0; x < kw; x++) {
								int ii = i * stride + y;
								int jj = j * stride + x;
								int in_idx = jj + iw * (ii + ih * (c + ch * n));
								if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
									out[out_idx] += pout[in_idx];
								}
							}
						}
                        if ((kh*kw & (kh*kw - 1)) == 0) // if power of 2
                        #if TRUNC_APPROACH == 0
                            out[out_idx] *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator); //TODO: Do shifts instead
                        #else
                            #if TRUNC_THEN_MULT == 0
                            out[out_idx] = out[out_idx].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator)); //TODO: Do shifts instead
                            #else
                                continue;
                            #endif
                        #endif
                        else 
                        #if TRUNC_APPROACH == 0
                            out[out_idx] *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator); //TODO: Do shifts instead
                        #else
                            #if TRUNC_THEN_MULT == 0
                            out[out_idx] = out[out_idx].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator)); //TODO: Do shifts instead
                            #else
                                continue;
                            #endif
                        #endif
					}
				}
			}
		}
        T::communicate();
    
        /* this->output.setZero(); */
		/* T* out = this->output.data(); */
		/* const T* pout = prev_out.data(); */
		/* auto denominator = kh * kw; */
		/* for (int n = 0; n < batch; n++) { */
			/* for (int c = 0; c < ch; c++) { */
				/* for (int i = 0; i < oh; i++) { */
					/* for (int j = 0; j < ow; j++) { */
						/* int out_idx = j + ow * (i + oh * (c + ch * n)); */
						/* for (int y = 0; y < kh; y++) { */
							/* for (int x = 0; x < kw; x++) { */
								/* int ii = i * stride + y; */
								/* int jj = j * stride + x; */
								/* int in_idx = jj + iw * (ii + ih * (c + ch * n)); */
								/* if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) { */
									/* out[out_idx] += pout[in_idx]; */
								/* } */
							/* } */
						/* } */
						/* out[out_idx] /= denominator; */
					/* } */
				/* } */
			/* } */
		/* } */
        // commented out by author
		/*for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				const float* im = prev_out.data() + ihw * (c + channels * n);
				im2col(im, 1, ih, iw, kh, stride, 0, im_col.data());
				output.row(c + channels * n) = im_col.colwise().mean();
			}
		}*/
#if TRUNC_APPROACH == 0
        for (int i = 0; i < this->output.size(); i++)
            out[i].complete_public_mult_fixed();
#else
    #if TRUNC_THEN_MULT == 1
        trunc_2k_in_place(out, this->output.size());
        T::communicate();
        for (int i = 0; i < this->output.size(); i++)
            out[i] = out[i].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator)); 
    #else
        trunc_2k_in_place(out, this->output.size());
        T::communicate();
    #endif
#endif
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cout << "PARTY " << PARTY <<  ": Time for Pooling: " << double(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()/1000000) << "s, Output Size: " << this->output.size() << std::endl; 
	}

    template<typename T>
	void AvgPool2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		T* pd = prev_delta.data();
		const T* d = this->delta.data();
		float denominator = kh * kw;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				for (int i = 0; i < oh; i++) {
					for (int j = 0; j < ow; j++) {
						int cur_idx = j + ow * (i + oh * (c + ch * n));
						for (int y = 0; y < kh; y++) {
							for (int x = 0; x < kw; x++) {
								int ii = y + stride * i;
								int jj = x + stride * j;
								int prev_idx = jj + iw * (ii + ih * (c + ch * n));
								if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
									/* pd[prev_idx] = d[cur_idx] / denominator; */
                                    pd[prev_idx] += d[cur_idx].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator)); //TODO: extend with different trunc methods
                                    /* pd[prev_idx] += d[cur_idx].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator)); */
							        	
                                }
							}
						}
					}
				}
			}
		}
        /* trunc_2k_in_place(pd, prev_delta.size()); */
	}

    template<typename T>
	void AvgPool2d<T>::zero_grad() { this->delta.setZero(); }

    template<typename T>
	vector<int> AvgPool2d<T>::output_shape() { return { batch, ch, oh, ow }; }
}

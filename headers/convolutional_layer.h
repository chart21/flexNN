#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Conv2d : public Layer<T>
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
        int stride;
        bool use_bias;
		string option;
		MatX<T> dkernel;
		VecX<T> dbias;
		MatX<T> im_col;
	public:
#if PUBLIC_WEIGHTS == 1
        MatX<UINT_TYPE> kernel;
        VecX<UINT_TYPE> bias;
#else
		MatX<T> kernel;
		VecX<T> bias;
#endif
		Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias = "true",
			string option = "kaiming_uniform");
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Conv2d<T>::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
        int stride,
		int padding,
        bool use_bias,
		string option
	) :
		Layer<T>(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
        stride(stride),
		pad(padding),
        use_bias(use_bias),
		option(option) {}

    template<typename T>
	void Conv2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, stride, pad);
		ow = calc_outsize(iw, kw, stride, pad);
		ohw = oh * ow;

		this->output.resize(batch * oc, ohw);
		this->delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
        if(use_bias)
        {
            bias.resize(oc);
            dbias.resize(oc);}
        else
        {
            bias.resize(0);
            dbias.resize(0);
        }

		im_col.resize(ic * kh * kw, ohw);

	    #if IS_TRAINING == 1	
		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
        init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
        #endif
	}

    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
        T::communicate();
        this->output.setZero();
#if TRUNC_DELAYED == 1
        
        if(delayed)
#if TRUNC_APPROACH == 0
            trunc_pr_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
#elif TRUNC_APPROACH == 1
            trunc_2k_in_place(const_cast<T*>(prev_out.data()), prev_out.size(),false);
#elif TRUNC_APPROACH == 2
            trunc_exact_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
#endif
        delayed = true;
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4 // Outsource whole convolution to GPU
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;

		    const T* im = prev_out.data() + (ic * ihw) * n;
            const T* W = kernel.data();
            int local_batch = 1;
            T::CONV_2D( im,W,C, local_batch, ih, iw, ic, oc, kh, kw, pad, stride, 1);
            send_GEMM_GPU(C, oc, ohw);
        }
#else // CPU or outsource only GEMM to GPU
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;
		    const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            auto A = kernel.data();
            #if USE_CUDA_GEMM == 0 //CPU uses transposed matrix
            MatX<T> BM = im_col.transpose();
            auto B = BM.data();
            #else
            auto B = im_col.data();
            #endif
            const int m = oc;
            const int p = ohw;
            const int f = kernel.cols();
            prepare_GEMM(A, B, C, m, p, f,true);
        }
#endif
    T::communicate();
    for (int n = 0; n < batch; n++) {
        auto C = this->output.data() + (oc * ohw) * n;
        complete_GEMM(C, oc, ohw);
    }
    
    #if TRUNC_DELAYED == 0 && TRUNC_APPROACH == 1
        trunc_2k_in_place(this->output.data(), this->output.size(),false);
    #elif TRUNC_DELAYED == 0 && TRUNC_APPROACH == 2
        trunc_exact_in_place(this->output.data(), this->output.size());
    #endif

if(use_bias)
{
    auto C = this->output.data();
    auto B = bias.data();
		for (int n = 0; n < batch; n++)
            for(int i = 0; i < oc; ++i)
                for(int j = 0; j < ohw; ++j)
                    add_bias(C[n*oc*ohw + i*ohw + j], B[i]);
}            
            
            





            }

    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		/* for (int n = 0; n < batch; n++) { */
		/* 	const T* im = prev_out.data() + (ic * ihw) * n; */
		/* 	im2col(im, ic, ih, iw, kh, 1, pad, im_col.data()); */
		/* 	dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose(); // TODO: change to prepare dot/ manual looping, no Eigen */
		/* 	dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum(); */
		/* } */

		/* if (!this->is_first) { */
		/* 	for (int n = 0; n < batch; n++) { */
		/* 		T* begin = prev_delta.data() + ic * ihw * n; */
		/* 		im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw); */
		/* 		col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin); */
		/* 	} */
		/* } */
	}

    template<typename T>
	void Conv2d<T>::update_weight(float lr, float decay)
	{
		/* float t1 = (1 - (2 * lr * decay) / batch); */
		/* float t2 = lr / batch; */

		/* if (t1 != 1) { */
		/* 	kernel *= t1; */
		/* 	bias *= t1; */
		/* } */

		/* kernel -= t2 * dkernel; */
		/* bias -= t2 * dbias; */
	}

    template<typename T>
	void Conv2d<T>::zero_grad()
	{
		this->delta.setZero();
		dkernel.setZero();
		dbias.setZero();
	}

    template<typename T>
	vector<int> Conv2d<T>::output_shape() { return { batch, oc, oh, ow }; }
}

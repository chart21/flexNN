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
        const int TILE_SIZE = 64;
        for(int i = 0; i < this->output.size(); ++i)
            this->output(i) = T(0);
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;
		const T* im = prev_out.data() + (ic * ihw) * n;
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
            const T* W = kernel.data();
            /* std::cout << "Y dimensinos:" << "n: " << 1 << " oc: " << oc << " ohw: " << ohw << "total: " << this->output.size() << std::endl; */
            int local_batch = 1;
            

            T::CONV_2D( im,W,C, local_batch, ih, iw, ic, oc, kh, kw, pad, stride, 1);
#else
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
#endif

#if USE_CUDA_GEMM == 0
            auto A = kernel.data();
            MatX<T> BM = im_col.transpose();
            auto B = BM.data();
            const int m = oc;
            const int p = ohw;
            const int f = kernel.cols();
  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    const int iip = ii*p;
                    const int iif = ii*f;
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                        const int jjf = jj*f;
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
#if PUBLIC_WEIGHTS == 0
                            temp += A[iif+kk].prepare_dot(B[jjf + kk]);
#else
                            temp += B[jjf + kk].mult_public(A[iif+kk]);
#endif
                        }
                        C[iip + jj] += temp;
                    }
                }
            }

            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                    C[row + jj].mask_and_send_dot_without_trunc();
#else
                    C[row + jj].mask_and_send_dot();
#endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    #else
                    C[row + jj].prepare_mult_public_fixed(1); //initiate truncation
    #endif
#endif
                    /* C[row + jj].mask_and_send_dot(); */
                }
            }
            /*     } */
            /* } */
        }
    }

}
/* for (int n = 0; n < batch; n++) { */
/*     auto C = this->output.data() + (oc * ohw) * n; */
/*   for (int i = 0; i < m; i += TILE_SIZE) { */
/*       int i_max = std::min(i + TILE_SIZE, m); */
/*       for (int j = 0; j < p; j += TILE_SIZE) { */
/*           int j_max = std::min(j + TILE_SIZE, p); */
/*             for (int ii = i; ii < i_max; ++ii) { */
/*                 int row = ii*p; */
/*                 for (int jj = j; jj < j_max; ++jj) { */
/*                     C[row + jj].mask_and_send_dot(); */
/*                 } */
/*             } */
/*             } */
/*             } */
/* } */
/* for(int i = 0; i < this->output.size(); ++i) */
/*     this->output(i).mask_and_send_dot(); */

            T::communicate();
for (int n = 0; n < batch; n++) {
    auto C = this->output.data() + (oc * ohw) * n;
            const int m = oc;
            const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
      int i_max = std::min(i + TILE_SIZE, m);
      for (int j = 0; j < p; j += TILE_SIZE) {
          int j_max = std::min(j + TILE_SIZE, p);
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    /* C[row + jj].complete_mult(); */
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                C[row+jj].complete_mult_without_trunc();
#else
                C[row+jj].complete_mult();
#endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    #else
                C[row+jj].complete_mult_public_fixed();
    #endif
#endif
                }
            }
            }
            }
}
#elif USE_CUDA_GEMM == 1 || USE_CUDA_GEMM == 3 // Use CUDA GEMM
auto A = kernel.data();
auto B = im_col.data();
int mul_m = kernel.rows();
int mul_n = im_col.cols();
int mul_k = kernel.cols();
                    
/* for(int i = 0; i < kernel.size(); ++i) */
/* { */
/*     DATATYPE val = kernel(i).get_p1(); */
/*     alignas(sizeof(DATTYPE)) UINT_TYPE tmp[BASE_DIV]; */
/*     unorthogonalize_arithmetic(&val, tmp,1); */
/*         if (tmp[0] != tmp[1] || tmp[0] != tmp[2] || tmp[0] != tmp[3]) */
/*         { */
/*             std::cout << "tmp[0] = " << tmp[0] << "\n"; */
/*             std::cout << "tmp[1] = " << tmp[1] << "\n"; */
/*             std::cout << "tmp[2] = " << tmp[2] << "\n"; */
/*         } */
/* } */



T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);
#endif

#if USE_CUDA_GEMM > 0
for(int j = 0; j < oc*ohw; ++j)
{
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    C[j].mask_and_send_dot_without_trunc();
#else
    C[j].mask_and_send_dot();
    #endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
    C[j].prepare_mult_public_fixed(1); //initiate truncation
#endif
#endif
}
}
T::communicate();
for (int n = 0; n < batch; n++) {
    auto C = this->output.data() + (oc * ohw) * n;
    for(int i = 0; i < oc*ohw; ++i)
    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    C[i].complete_mult_without_trunc();
#else
    C[i].complete_mult();
#endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
    C[i].complete_mult_public_fixed();
#endif
#endif
}
}


#endif
/* for(int i = 0; i < this->output.size(); ++i) */
/*     this->output(i).complete_mult(); */

    /* for (int n = 0; n < batch; n++) { */
    /* auto C = this->output.data() + (oc * ohw) * n; */
  /* for (int i = 0; i < m; i += TILE_SIZE) { */
    /*   /1* _mm_prefetch(A + i * f, _MM_HINT_T0); *1/ */
    /*     int i_max = std::min(i + TILE_SIZE, m); */
    /*     for (int j = 0; j < p; j += TILE_SIZE) { */
    /*         /1* _mm_prefetch(B + j * f, _MM_HINT_T0); *1/ */
    /*         int j_max = std::min(j + TILE_SIZE, p); */
    /*             for (int ii = i; ii < i_max; ++ii) { */
    /*                 /1* const int row2 = ii*f+kk; *1/ */
    /*                 for (int jj = j; jj < j_max; ++jj) { */
    /*             /1* this->output(i) += bias(i % oc); // replace lower code *1/ */
    /*         } */
    /*             } */
    /*     } */
  /* } */
    /* } */
#if TRUNC_DELAYED == 0 && TRUNC_APPROACH == 1
        trunc_2k_in_place(this->output.data(), this->output.size());
#endif

if(use_bias)
{
#if TRUNC_DELAYED == 0
		for (int n = 0; n < batch; n++)
            this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
#else
        // multiply each bias by 2^FRACTIONAL
#if PUBLIC_WEIGHTS == 0
        std::transform(bias.data(), bias.data() + bias.size(), bias.data(), [](T x) { return x.mult_public(UINT_TYPE(1) << FRACTIONAL); });
		for (int n = 0; n < batch; n++)
            this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
#else
		for (int n = 0; n < batch; n++)
        {
            for(int i = 0; i < oc; ++i)
                for(int j = 0; j < ohw; ++j)
                    this->output(oc * n + i, j) += bias(i) << FRACTIONAL;
        }
#endif
#endif
}            /* for(int i = 0; i < oc; ++i) */ 
            /*     for(int j = 0; j < ohw; ++j) */ 
            /*         this->output(oc * n + i, j) += bias(i); */



#if SIMULATE_QUANT == 1
        for(int i = 0; i < this->output.size(); ++i)
        {
            this->output(i) = this->output(i).prepare_dot(this->output(i)); //simulate scale multiplication
            this->output(i).mask_and_send_dot();
        }
        T::communicate();
        for(int i = 0; i < this->output.size(); ++i)
            this->output(i).complete_mult();
#endif


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

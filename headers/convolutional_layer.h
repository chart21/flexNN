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
		MatX<T> im_col_t;  // CPU GEMM: the column matrix transposed, ohw x (ic * kh * kw)
		bool fuse_batchnorm_parameters;
	public:
		bool bake_output = true;  // feeds a baked ReLU directly: masks from the A2B bake (g_conv_bake, set in compile)
		int residual_producer_k = -1;  // the output is the other addend of residual sum k (g_res_producer_k)
		bool merge_skip = false;     // residual merge: a sum's addend computed first, not sent (g_res_merge_skip)
		bool merge_partner = false;  // residual merge: a sum's addend computed last, sends both (g_res_merge_add)
#if PUBLIC_WEIGHTS == 1
        MatX<UINT_TYPE> kernel;
        VecX<UINT_TYPE> bias;
#else
		MatX<T> kernel;
		VecX<T> bias;
#endif
		Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias = "true",
			string option = "kaiming_uniform");
		void enable_batchnorm_fusion();
		bool batchnorm_fusion_enabled() const;
		void enable_bias();
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
		fuse_batchnorm_parameters(false),
		option(option) {}

	template<typename T>
	void Conv2d<T>::enable_batchnorm_fusion()
	{
		fuse_batchnorm_parameters = true;
	}

	template<typename T>
	bool Conv2d<T>::batchnorm_fusion_enabled() const
	{
		return fuse_batchnorm_parameters;
	}

	template<typename T>
	void Conv2d<T>::enable_bias()
	{
		use_bias = true;
		if (bias.size() != oc) {
			bias.resize(oc);
			bias.setZero();
		}
		if (dbias.size() != oc) {
			dbias.resize(oc);
			dbias.setZero();
		}
	}

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

#if USE_CUDA_GEMM != 0
		im_col.resize(ic * kh * kw, ohw);
#endif

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
#if PROTOCOL == 4 && A_KNOWN == 0 && PUBLIC_WEIGHTS == 0 && BEAVER == 1
        // First layer: the raw data-owner input (m = 0 under SHARE_PREP) breaks the SecureML
        // truncation share-pair distribution - re-randomize it first (see remask_range in GEMM.hpp).
        if (this->is_first)
            remask_range(const_cast<T*>(prev_out.data()), (int)prev_out.size());
#endif
#if PUBLIC_WEIGHTS == 1
        // Only the network's first layer sees the raw data-owner input (non-owner mask = 0); route its truncation
        // to the *_a_known variant. Re-set every layer so later convs use the normal truncation.
        g_a_known_input = this->is_first ? 1 : 0;
#endif
        T::communicate();
#if ADDITIONAL_GEMM_THREADS > 0
        {  // the outputs are many (multi-batch: all lanes): zero them on the GEMM threads
            T* out = this->output.data();
            const size_t n_out = (size_t) this->output.size();
            constexpr int parts = ADDITIONAL_GEMM_THREADS + 1;
            GemmPool::get().run([&](int t) {
                for (size_t i = n_out * t / parts; i < n_out * (t + 1) / parts; i++)
                    out[i] = T(0);
            });
        }
#else
        this->output.setZero();
#endif
#if TRUNC_DELAYED == 1
        
        if(delayed)
#if TRUNC_APPROACH == 0
            trunc_pr_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
#elif TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4
            trunc_2k_in_place(const_cast<T*>(prev_out.data()), prev_out.size(),all_positive);
#elif TRUNC_APPROACH == 2
            trunc_exact_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
#elif TRUNC_APPROACH == 3
            trunc_exact_opt_in_place(const_cast<T*>(prev_out.data()), prev_out.size(),all_positive);
#endif
        delayed = true;
#endif


#if TRUNC_APPROACH > 0
    all_positive = false;
#endif

#if CHEETAH_CONV_EARLY_ACTIVE
        if (g_mask_pass)
        {
            // the mask-only forward: the conv triple's inputs, for the conv triples that start before the OT phase;
            // the output masks do not matter (the next conv's inputs come out of a ReLU, or the pass's check fails)
            T::RecordConv2dInputs(prev_out.data(), kernel.data(), batch, ih, iw, ic, oc, kh, kw);
            return;
        }
#endif
#if PROTOCOL == 4 && CONV_TRIPLES == 1 && PUBLIC_WEIGHTS == 0
        T::SetupConv2dTriples(prev_out.data(), kernel.data(), this->output.data(), batch, ih , iw, ic, oc, kh, kw, pad, stride, oh, ow);
#endif
#if PROTOCOL == 4 && BN2D_TRIPLES == 1 && PUBLIC_WEIGHTS == 0 && FUSE_CONV_BN_SIM == 1
        const auto move_var = new T[ch]; 
        // TODO: for fused conv-bn C needs to be set to beta - (sigma * mu) by modelowner and move_var needs to be set to the actual sigma array
        T::SetupBatchNorm2DTriples(prev_out.data(), move_var, this->output.data(), batch, ch, h, w);
        T::SetupBatchNorm2DTriples(prev_out.data(), move_var, this->output.data(), batch, ch, h, w);
        T::SetupBatchNorm2DTriples(prev_out.data(), move_var, this->output.data(), batch, ch, h, w);
        delete[] move_var;
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
#if PROTOCOL == 4 && ROT_PREPROCESSING_OPT == 1 && \
    ((RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1) || A2B_CONV_BAKE_ACTIVE)
        // see fully_connected_layer.h: publish the effective bias mask (expanded per output value,
        // bias repeats per channel) so the reshare bake can pre-compensate the post-GEMM bias add
        std::vector<DATATYPE> bake_bias_l;  // one mask per channel, repeated ohw times (g_bake_bias_rep)
        if (use_bias)
        {
            bake_bias_l.resize((size_t)oc);
            for (int i = 0; i < oc; ++i)
            {
#if PUBLIC_WEIGHTS == 1
                DATATYPE bl = SET_ALL_ZERO();  // public bias is a plain value and carries no mask
#elif TRUNC_DELAYED == 0
                DATATYPE bl = bias.data()[i].get_share().get_mask();
#else
                DATATYPE bl = bias.data()[i].mult_public(UINT_TYPE(1) << FRACTIONAL).get_share().get_mask();
#endif
                bake_bias_l[(size_t)i] = bl;
            }
            g_bake_bias_l = bake_bias_l.data();
            g_bake_bias_len = (uint64_t)oc * ohw;
            g_bake_bias_rep = (uint64_t)ohw;
        }
#endif
        g_conv_bake = bake_output;
        g_res_producer_k = residual_producer_k;
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;
		    const T* im = prev_out.data() + (ic * ihw) * n;
            auto A = kernel.data();
#if GEMM_FAST_CONV_GPU && USE_CUDA_GEMM == 0
            // GEMM_FAST_GPU: the product with its column matrix built on the GPU (only the input travels); the
            // prepare_GEMM below then only masks and sends
            if (gemm_fast::accumulate_conv(A, im, C, oc, ic, ih, iw, kh, stride, pad)) {
                g_bake_batch_offset = (uint64_t)(oc * ohw) * n;
                prepare_GEMM(A, (T*) nullptr, C, oc, ohw, (int) kernel.cols(), true);
                continue;
            }
#endif
            #if USE_CUDA_GEMM == 0 //CPU uses transposed matrix, built directly (on the GEMM threads)
            if (im_col_t.rows() != ohw)
                im_col_t.resize(ohw, ic * kh * kw);
            T* B = im_col_t.data();
#if ADDITIONAL_GEMM_THREADS > 0
            if ((size_t)ohw * ic * kh * kw >= 16384) {
                const int parts = ADDITIONAL_GEMM_THREADS + 1;
                GemmPool::get().run([&](int t) {
                    im2col_transposed(im, ic, ih, iw, kh, stride, pad, B, (int)((long)ohw * t / parts),
                                      (int)((long)ohw * (t + 1) / parts));
                });
            } else
#endif
                im2col_transposed(im, ic, ih, iw, kh, stride, pad, B, 0, ohw);
            #else
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            auto B = im_col.data();
            #endif
            const int m = oc;
            const int p = ohw;
            const int f = kernel.cols();
            g_bake_batch_offset = (uint64_t)(oc * ohw) * n;  // batch-global base for the reshare bake
            prepare_GEMM(A, B, C, m, p, f,true);
        }
        g_bake_batch_offset = 0;
        g_conv_bake = true;
        g_res_producer_k = -1;
#if PROTOCOL == 4 && ROT_PREPROCESSING_OPT == 1 && \
    ((RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1) || A2B_CONV_BAKE_ACTIVE)
        g_bake_bias_l = nullptr;
        g_bake_bias_len = 0;
        g_bake_bias_rep = 1;
#endif
#endif
    T::communicate();
    for (int n = 0; n < batch; n++) {
        auto C = this->output.data() + (oc * ohw) * n;
        complete_GEMM(C, oc, ohw);
    }
    
    #if TRUNC_DELAYED == 0 && (TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4)
        trunc_2k_in_place(this->output.data(), this->output.size(),false);
    #elif TRUNC_DELAYED == 0 && TRUNC_APPROACH == 2
        trunc_exact_in_place(this->output.data(), this->output.size());
    #elif TRUNC_DELAYED == 0 && TRUNC_APPROACH == 3
        trunc_exact_opt_in_place(this->output.data(), this->output.size());
    #endif

if(use_bias)
{
    auto C = this->output.data();
    auto B = bias.data();
#if ADDITIONAL_GEMM_THREADS > 0
    constexpr int parts = ADDITIONAL_GEMM_THREADS + 1;
    const int rows = batch * oc;  // (image, channel) rows of ohw outputs, on the GEMM threads
    GemmPool::get().run([&](int t) {
        for (int r = rows * t / parts; r < rows * (t + 1) / parts; ++r)
            for (int j = 0; j < ohw; ++j)
                add_bias(C[(size_t)r * ohw + j], B[r % oc]);
    });
#else
		for (int n = 0; n < batch; n++)
            for(int i = 0; i < oc; ++i)
                for(int j = 0; j < ohw; ++j)
                    add_bias(C[n*oc*ohw + i*ohw + j], B[i]);
#endif
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

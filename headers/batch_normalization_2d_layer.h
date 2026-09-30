#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class BatchNorm2d : public Layer<T>
	{
	private:
		int batch;
		int ch;
		int h;
		int w;
		int hw;
		float eps;
		float momentum;
		VecX<T> mu;
		VecX<T> var;
		VecX<T> dgamma;
		VecX<T> dbeta;
		VecX<T> sum1;
		VecX<T> sum2;
	public:
		bool bake_output = true;  // feeds a baked ReLU directly (see Conv2d::bake_output; BN_BAKE_SUPPORTED)
		MatX<T> xhat;
		MatX<T> dxhat;
#if PUBLIC_WEIGHTS == 1
        VecX<UINT_TYPE> move_mu;
        VecX<UINT_TYPE> move_var;
        VecX<UINT_TYPE> gamma;
        VecX<UINT_TYPE> beta;
#else
		VecX<T> move_mu;
		VecX<T> move_var;
		VecX<T> gamma;
		VecX<T> beta;
#endif
		BatchNorm2d(float eps = 0.00001f, float momentum = 0.9f);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	private:
		void calc_batch_mu(const MatX<T>& prev_out);
		void calc_batch_var(const MatX<T>& prev_out);
		void normalize_and_shift(const MatX<T>& prev_out, bool is_training);
	};

    template<typename T>
	BatchNorm2d<T>::BatchNorm2d(float eps, float momentum) :
		Layer<T>(LayerType::BATCHNORM2D),
		batch(0),
		ch(0),
		h(0),
		w(0),
		hw(0),
		eps(eps),
		momentum(momentum) {}

    template<typename T>
	void BatchNorm2d<T>::set_layer(const vector<int>& input_shape)
	{
		assert(input_shape.size() == 4 && "BatchNorm2d::set_layer(const vector<int>&): Must be followed by 2d layer.");

		batch = input_shape[0];
		ch = input_shape[1];
		h = input_shape[2];
		w = input_shape[3];
		hw = h * w;

		this->output.resize(batch * ch, hw);
		this->delta.resize(batch * ch, hw);
		xhat.resize(batch * ch, hw);
		dxhat.resize(batch * ch, hw);
		move_mu.resize(ch);
		move_var.resize(ch);
		mu.resize(ch);
		var.resize(ch);
		gamma.resize(ch);
		dgamma.resize(ch);
		beta.resize(ch);
		dbeta.resize(ch);
		sum1.resize(ch);
		sum2.resize(ch);
#if IS_TRAINING == 1
		move_mu.setZero();
		move_var.setZero();
		gamma.setConstant(1.f);
		beta.setZero();
#endif
	}

    template<typename T>
	void BatchNorm2d<T>::forward(const MatX<T>& prev_out, bool is_training)
    {
#if CHEETAH_CONV_EARLY_ACTIVE
        if (g_mask_pass)
        {
            this->output.setZero();  // the mask-only forward: a ReLU follows, whose outputs have committed masks
            return;
        }
#endif

#if IS_TRAINING == 1
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, is_training);
			// update moving mu and var
			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
#else
#if FUSE_CONV_BN_SIM == 1 || FUSE_CONV_BN == 1
            std::copy(prev_out.data(), prev_out.data() + prev_out.size(), this->output.data());
            return;
#endif
#if TRUNC_DELAYED == 1
        if(delayed)
#if TRUNC_APPROACH == 0
#if PROTOCOL == 4 && PUBLIC_WEIGHTS == 1 && DATAOWNER != -1
        {
            if (g_a_known_input)  // the first conv's output: data-owner sharing, truncated exactly by the owner
                trunc_a_known_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
            else
                trunc_pr_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
        }
#else
            trunc_pr_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
#endif
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

#if PROTOCOL == 4 && BN2D_TRIPLES == 1 && PUBLIC_WEIGHTS == 0
        // The triple is taken over the masks of the product's operands (after the delayed truncation above, which
        // re-masks the input). With A_KNOWN=0 it multiplies the FULL masks, and the operand is x - mu, whose mask
        // includes mu's (the model owner's) mask: using x's mask left out lambda_s * lambda_mu = s * mu. With
        // A_KNOWN=1 only the data owner's mask share enters, which mu does not change.
#if A_KNOWN == 0
        {
            MatX<T> centered(prev_out.rows(), prev_out.cols());
            for (int n = 0; n < batch; n++)
                for (int c = 0; c < ch; c++)
                    for (int j = 0; j < hw; j++)
                        centered(c + ch * n, j) = prev_out(c + ch * n, j) - move_mu[c];
            T::SetupBatchNorm2DTriples(centered.data(), move_var.data(), this->output.data(), batch, ch, h, w);
        }
#else
        T::SetupBatchNorm2DTriples(prev_out.data(), move_var.data(), this->output.data(), batch, ch, h, w);
#endif
#endif

			normalize_and_shift(prev_out, is_training);
#endif
	}

    template<typename T>
	void BatchNorm2d<T>::calc_batch_mu(const MatX<T>& prev_out)
	{
		mu.setZero();
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				mu[c] += prev_out.row(c + ch * n).mean() / batch;
			}
		}
	}

    template<typename T>
	void BatchNorm2d<T>::calc_batch_var(const MatX<T>& prev_out)
	{
		var.setZero();
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				float m = mu[c];
				float v = 0.f;
				for (int j = 0; j < hw; j++) {
					float diff = prev_out(i, j) - m;
					v += diff * diff;
				}
				var[c] += v / hw / batch;
			}
		}
	}


    template<typename T>
	void BatchNorm2d<T>::normalize_and_shift(const MatX<T>& prev_out, bool is_training)
	{
        #if IS_TRAINING == 1
        const T* M = mu.data();
        const T* V = var.data();
        #else
        const auto* M = move_mu.data();
        const auto* V = move_var.data();
        #endif
#if BN_BAKE_SUPPORTED
        // like a conv's bias (convolutional_layer.h): beta is added after the multiplication, so the bakes pre-compensate
        // its mask; only when this layer feeds a baked ReLU directly may it take the committed masks
        std::vector<DATATYPE> bake_beta_l((size_t) ch * hw);
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < hw; j++) bake_beta_l[(size_t) c * hw + j] = beta[c].get_share().get_mask();
        g_bake_bias_l = bake_beta_l.data();
        g_bake_bias_len = (uint64_t) ch * hw;
        g_bake_batch_offset = 0;
        g_conv_bake = bake_output;
#endif
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				auto m = M[c];
                auto s = V[c];
				for (int j = 0; j < hw; j++) {
#if PUBLIC_WEIGHTS == 0
#if PROTOCOL == 4 && BN2D_TRIPLES == 1
#if A_KNOWN == 1
                this->output(i, j) = s.prepare_dot_ex_lxly_a_known(prev_out(i, j) - m);
#else
                this->output(i, j) = s.prepare_dot_ex_lxly(prev_out(i, j) - m);  // AB-flavored BN triples
#endif
#else

#if A_KNOWN == 1 
                this->output(i, j) = s.prepare_dot_a_known(prev_out(i, j) - m);
#else
                this->output(i, j) = s.prepare_dot(prev_out(i, j) - m);
#endif					
#endif
                
#if TRUNC_APPROACH > 0 || TRUNC_DELAYED == 1
#if PROTOCOL == 4 && BN2D_TRIPLES == 1
                    this->output(i, j).mask_and_send_dot_without_trunc_with_triple();
#else
                    this->output(i, j).mask_and_send_dot_without_trunc();
#endif
#else
#if BN_BAKE_SUPPORTED
                    this->output(i, j).mask_and_send_dot_with_triple(i * hw + j);  // indexed: the bakes (NCHW index)
#elif PROTOCOL == 4 && BN2D_TRIPLES == 1
                    this->output(i, j).mask_and_send_dot_with_triple();
#else
                    this->output(i, j).mask_and_send_dot();
#endif
#endif
#else
#if TRUNC_APPROACH > 0 || TRUNC_DELAYED == 1
                    this->output(i, j) = (prev_out(i, j) - m).mult_public(s);
#else
                    this->output(i, j) = (prev_out(i, j) - m) * s;
#endif
#endif
				}
			}
		}
#if BN_BAKE_SUPPORTED
        g_bake_bias_l = nullptr;
        g_bake_bias_len = 0;
        g_conv_bake = true;
#endif
        T::communicate();
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				for (int j = 0; j < hw; j++) {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_APPROACH > 0 || TRUNC_DELAYED == 1
                    this->output(i, j).complete_mult_without_trunc();
#else
					this->output(i, j).complete_mult();
                    this->output(i, j) += beta[c];
#endif
#else
#if TRUNC_APPROACH > 0 || TRUNC_DELAYED == 1
                    // do nothing
#else
                    this->output(i, j).complete_public_mult_fixed();
                    this->output(i, j) += beta[c];
#endif
#endif

                }
			}
		}
        T::communicate();

#if TRUNC_APPROACH > 0 && TRUNC_DELAYED == 0
#if TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4
        trunc_2k_in_place(this->output.data(), this->output.size(),false);
#elif TRUNC_APPROACH == 2
        trunc_exact_in_place(this->output.data(), this->output.size());
#elif TRUNC_APPROACH == 3
        trunc_exact_opt_in_place(this->output.data(), this->output.size());
#endif
#endif
    

#if  TRUNC_APPROACH > 0 || TRUNC_DELAYED == 1
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				for (int j = 0; j < hw; j++) 
                {
                    add_bias(this->output(i,j),beta[c]);

                }

            }}
    #endif

	}



    template<typename T>
	void BatchNorm2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
#if IS_TRAINING == 1
		// calc dxhat
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				float g = gamma[c];
				for (int j = 0; j < hw; j++) {
					dxhat(i, j) = this->delta(i, j) * g;
				}
			}
		}

		// calc Sum(dxhat), Sum(dxhat * xhat)
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				float s1 = 0.f;
				float s2 = 0.f;
				for (int j = 0; j < hw; j++) {
					s1 += dxhat(i, j);
					s2 += dxhat(i, j) * xhat(i, j);
				}
				sum1[c] += s1 / hw;
				sum2[c] += s2 / hw;
			}
		}

		// calc dx, dgamma, dbeta
		float m = (float)batch;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				float s1 = sum1[c];
				float s2 = sum2[c];
				float dg = 0.f;
				float db = 0.f;
				float denominator = m * std::sqrt(var[c] + eps);
				for (int j = 0; j < hw; j++) {
					prev_delta(i, j) = (m * dxhat(i, j)) - s1 - (xhat(i, j) * s2);
					prev_delta(i, j) /= denominator;
					dg += (xhat(i, j) * this->delta(i, j));
					db += this->delta(i, j);
				}
				dgamma[c] += dg;
				dbeta[c] += db;
			}
		}
#endif
	}

    template<typename T>
	void BatchNorm2d<T>::update_weight(float lr, float decay)
	{
#if IS_TRAINING == 1
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		if (t1 != 1) {
			gamma *= t1;
			beta *= t1;
		}
		gamma -= t2 * dgamma;
		beta -= t2 * dbeta;
#endif
	}
    template<typename T>
	void BatchNorm2d<T>::zero_grad()
	{
		this->delta.setZero();
		dxhat.setZero();
		dgamma.setZero();
		dbeta.setZero();
		sum1.setZero();
		sum2.setZero();
	}

    template<typename T>
	vector<int> BatchNorm2d<T>::output_shape() { return { batch, ch, h, w }; }
}

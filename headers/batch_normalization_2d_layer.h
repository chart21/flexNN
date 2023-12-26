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
#if IS_TRAINING == 1
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, is_training);
			// update moving mu and var
			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
#else
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
		/* const T* M = mu.data(); */
		/* const T* V = var.data(); */

		/* if (!is_training) { */
		/* 	M = move_mu.data(); */
		/* 	V = move_var.data(); */
		/* } */
        #if IS_TRAINING == 1
        const T* M = mu.data();
        const T* V = var.data();
        #else
        const auto* M = move_mu.data();
        const auto* V = move_var.data();
        #endif
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				auto m = M[c];
				/* float s = std::sqrt(V[c] + eps); */
                auto s = V[c];
				/* T g = gamma[c]; */
				/* T b = beta[c]; */
				for (int j = 0; j < hw; j++) {
#if PUBLIC_WEIGHTS == 0
					xhat(i, j) = (prev_out(i, j) - m).prepare_dot(s);
                    xhat(i, j).mask_and_send_dot();
#else
                    xhat(i, j) = (prev_out(i, j) - m) * s;
#endif
				}
			}
		}
        T::communicate();
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				/* T m = M[c]; */
				/* float s = std::sqrt(V[c] + eps); */
                /* T s = V[c]; */
				auto g = gamma[c];
				/* T b = beta[c]; */
				for (int j = 0; j < hw; j++) {
#if PUBLIC_WEIGHTS == 0
					xhat(i, j).complete_mult();
					this->output(i, j) = g.prepare_dot(xhat(i, j));
                    this->output(i, j).mask_and_send_dot();
#else
                    xhat(i, j).complete_public_mult_fixed();
                    this->output(i, j) = xhat(i, j) * g;
#endif
				}
			}
		}
        T::communicate();
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				/* T m = M[c]; */
				/* float s = std::sqrt(V[c] + eps); */
                /* T s = V[c]; */
				/* T g = gamma[c]; */
				auto b = beta[c];
				for (int j = 0; j < hw; j++) {
#if PUBLIC_WEIGHTS == 0
					this->output(i, j).complete_mult();
#else
                    this->output(i, j).complete_public_mult_fixed();
#endif
                    this->output(i, j) += b;
				}
			}
		}
	}


    /* template<typename T> */
	/* void BatchNorm2d<T>::normalize_and_shift(const MatX<T>& prev_out, bool is_training) */
	/* { */
/* #if IS_TRAINING == 1 */
		/* const T* M = mu.data(); */
		/* const T* V = var.data(); */
/* #else */
    /*     const T* M = move_mu.data(); */
    /*     const T* V = move_var.data(); */
/* #endif */

		/* for (int n = 0; n < batch; n++) { */
			/* for (int c = 0; c < ch; c++) { */
				/* int i = c + ch * n; */
				/* T m = M[c]; */
				/* T s = V[c]; */
				/* for (int j = 0; j < hw; j++) { */
					/* xhat(i, j) = (prev_out(i, j) - m).prepare_dot(s); */
/* #if TRUNC_APPROACH == 0 */
    /*                 xhat(i, j).mask_and_send_dot(); */
/* #else */
    /*                 xhat(i, j).mask_and_send_dot_without_trunc(); */
/* #endif */
				/* } */
			/* } */
		/* } */
    /*     T::communicate(); */
		/* for (int n = 0; n < batch; n++) { */
			/* for (int c = 0; c < ch; c++) { */
				/* int i = c + ch * n; */
				/* T g = gamma[c]; */
				/* for (int j = 0; j < hw; j++) { */
/* #if TRUNC_APPROACH == 0 */
					/* xhat(i, j).complete_mult(); */
					/* this->output(i, j) = g.prepare_dot(xhat(i, j)); */
    /*                 this->output(i, j).mask_and_send_dot(); */
/* #else */
    /*                 xhat(i, j).complete_mult_without_trunc(); */
/* #endif */
    /*             } */
    /*         } */
    /*     } */
/* #if TRUNC_APPROACH == 1 */
    /*     trunc_2k_in_place(xhat.data(), batch * ch * hw); */
    /*     for (int n = 0; n < batch; n++) { */
    /*         for (int c = 0; c < ch; c++) { */
    /*             int i = c + ch * n; */
    /*             T g = gamma[c]; */
    /*             for (int j = 0; j < hw; j++) { */
    /*                 this->output(i, j) = g.prepare_dot(xhat(i, j)); */
    /*                 this->output(i, j).mask_and_send_dot_without_trunc(); */
    /*             } */
    /*         } */
    /*     } */
/* #endif */

    /*     T::communicate(); */
		/* for (int n = 0; n < batch; n++) { */
			/* for (int c = 0; c < ch; c++) { */
				/* int i = c + ch * n; */
				/* T b = beta[c]; */
				/* for (int j = 0; j < hw; j++) { */
/* #if TRUNC_APPROACH == 0 */
    /*                 this->output(i, j).complete_mult(); */
    /*                 this->output(i, j) += b; */
/* #else */
    /*                 this->output(i, j).complete_mult_without_trunc(); */
/* #endif */
    /*             } */
    /*         } */
    /*     } */
/* #if TRUNC_APPROACH == 1 */
    /*     trunc_2k_in_place(this->output.data(), batch * ch * hw); */
    /*     for (int n = 0; n < batch; n++) { */
    /*         for (int c = 0; c < ch; c++) { */
    /*             int i = c + ch * n; */
    /*             T b = beta[c]; */
    /*             for (int j = 0; j < hw; j++) { */
    /*                 this->output(i, j) += b; */
    /*             } */
    /*         } */
    /*     } */
/* #endif */
    /* } */


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

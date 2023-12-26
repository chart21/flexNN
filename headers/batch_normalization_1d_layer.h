#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class BatchNorm1d : public Layer<T>
	{
	private:
		int batch;
		int n_feat;
		float eps;
		float momentum;
		MatX<T> xhat;
		MatX<T> dxhat;
		RowVecX<T> mu;
		RowVecX<T> var;
		RowVecX<T> dgamma;
		RowVecX<T> dbeta;
		RowVecX<T> sum1;
		RowVecX<T> sum2;
	public:
#if PUBLIC_WEIGHTS == 1
        RowVecX<UINT_TYPE> move_mu;
        RowVecX<UINT_TYPE> move_var;
        RowVecX<UINT_TYPE> gamma;
        RowVecX<UINT_TYPE> beta;
#else
		RowVecX<T> move_mu;
		RowVecX<T> move_var;
		RowVecX<T> gamma;
		RowVecX<T> beta;
#endif
		BatchNorm1d(float eps = 0.00001f, float momentum = 0.9f);
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
	BatchNorm1d<T>::BatchNorm1d(float eps, float momentum) :
		Layer<T>(LayerType::BATCHNORM1D),
		batch(0),
		n_feat(0),
		eps(eps),
		momentum(momentum) {}

    template<typename T>
	void BatchNorm1d<T>::set_layer(const vector<int>& input_shape)
	{
		assert(input_shape.size() == 2 && "BatchNorm1d::set_layer(const vector<int>&): Must be followed by Linear layer.");

		batch = input_shape[0];
		n_feat = input_shape[1];

		this->output.resize(batch, n_feat);
		this->delta.resize(batch, n_feat);
		xhat.resize(batch, n_feat);
		dxhat.resize(batch, n_feat);
		move_mu.resize(n_feat);
		move_var.resize(n_feat);
		mu.resize(n_feat);
		var.resize(n_feat);
		gamma.resize(n_feat);
		dgamma.resize(n_feat);
		beta.resize(n_feat);
		dbeta.resize(n_feat);
		sum1.resize(n_feat);
		sum2.resize(n_feat);
        #if IS_TRAINING == 1
		move_mu.setZero();
		move_var.setZero();
		gamma.setConstant(1.f);
		beta.setZero();
        #endif
	}

    template<typename T>
	void BatchNorm1d<T>::forward(const MatX<T>& prev_out, bool is_training)
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
	void BatchNorm1d<T>::calc_batch_mu(const MatX<T>& prev_out)
	{
		mu = prev_out.colwise().mean();
	}

    template<typename T>
	void BatchNorm1d<T>::calc_batch_var(const MatX<T>& prev_out)
	{
		var.setZero();
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				float diff = prev_out(i, j) - mu[j];
				var[j] += diff * diff / batch; // (batch - 1)?
			}
		}
	}

    template<typename T>
	void BatchNorm1d<T>::normalize_and_shift(const MatX<T>& prev_out, bool is_training)
	{
#if IS_TRAINING == 1
		const T* M = mu.data();
		const T* V = var.data();
#else
        const auto* M = move_mu.data();
        const auto* V = move_var.data();
#endif
		/* if (!is_training) { */
		/* 	M = move_mu.data(); */
		/* 	V = move_var.data(); */
		/* } */

		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
#if PUBLIC_WEIGHTS == 0
				xhat(i, j) = (prev_out(i, j) - M[j]).prepare_dot( V[j]);
                xhat(i, j).mask_and_send_dot();
#else
                xhat(i, j) = (prev_out(i, j) - M[j]) * V[j];
#endif
			}
		}
        T::communicate();
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
#if PUBLIC_WEIGHTS == 0
				xhat(i, j).complete_mult();
				this->output(i, j) = gamma[j].prepare_dot(xhat(i, j));
                this->output(i, j).mask_and_send_dot();
#else
                xhat(i, j).complete_public_mult_fixed();
                this->output(i, j) = xhat(i, j) * gamma[j];
#endif
			}
		}
        T::communicate();
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
#if PUBLIC_WEIGHTS == 0
				this->output(i, j).complete_mult();
#else
                this->output(i, j).complete_public_mult_fixed();
#endif
                this->output(i, j) += beta[j];
			}
		}

	}


    /* template<typename T> */
	/* void BatchNorm1d<T>::normalize_and_shift(const MatX<T>& prev_out, bool is_training) */
	/* { */
/* #if IS_TRAINING == 1 */
		/* const T* M = mu.data(); */
		/* const T* V = var.data(); */
/* #else */
    /*     const T* M = move_mu.data(); */
    /*     const T* V = move_var.data(); */
/* #endif */

		/* for (int i = 0; i < batch; i++) { */
			/* for (int j = 0; j < n_feat; j++) { */
    /*             xhat(i, j) = (prev_out(i, j) - M[j]).prepare_dot(V[j]); */
/* #if TRUNC_APPROACH == 0 */
    /*             xhat(i, j).mask_and_send_dot(); */
/* #else */
    /*             xhat(i, j).mask_and_send_dot_without_trunc(); */
/* #endif */
				/* /1* xhat(i, j) = (prev_out(i, j) - M[j]) / std::sqrt(V[j] + eps); *1/ */
				/* /1* this->output(i, j) = gamma[j] * xhat(i, j) + beta[j]; *1/ */
			/* } */
		/* } */
    /*     T::communicate(); */
/* #if TRUNC_APPROACH == 0 */
		/* for (int i = 0; i < batch; i++) { */
			/* for (int j = 0; j < n_feat; j++) { */
    /*         xhat(i, j).complete_mult(); */
			/* this->output(i, j) = gamma[j].prepare_dot(xhat(i, j)); */
    /*         this->output(i, j).mask_and_send_dot(); */
    /*     } */
	/* } */
/* #else */
		/* for (int i = 0; i < batch; i++) { */
			/* for (int j = 0; j < n_feat; j++) { */
    /*         xhat(i, j).complete_mult_without_trunc(); */
    /*     } */
	/* } */
    /*     trunc_2k_in_place(xhat.data(), batch * n_feat); */
    /*     for (int i = 0; i < batch; i++) { */
    /*         for (int j = 0; j < n_feat; j++) { */
    /*         this->output(i, j) = gamma[j].prepare_dot(xhat(i, j)); */
    /*         this->output(i, j).mask_and_send_dot_without_trunc(); */
    /*     } */
    /*     } */
/* #endif */
    /*     T::communicate(); */
		/* for (int i = 0; i < batch; i++) { */
			/* for (int j = 0; j < n_feat; j++) { */
    /* #if TRUNC_APPROACH == 0 */
    /*         this->output(i, j).complete_mult(); */
    /*         this->output(i, j) += beta[j]; */
    /* #else */
    /*         this->output(i, j).complete_mult_without_trunc(); */
    /* #endif */

    /*     } */
    /*     } */
    /* #if TRUNC_APPROACH == 1 */
    /* trunc_2k_in_place(this->output.data(), batch * n_feat); */
    /* for (int i = 0; i < batch; i++) { */
    /*     for (int j = 0; j < n_feat; j++) { */
    /*         this->output(i, j) += beta[j]; */
    /*     } */
    /* } */
    /* #endif */
            

/* } */

    template<typename T>
	void BatchNorm1d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
#if IS_TRAINING == 1
		// calc dxhat
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				dxhat(i, j) = this->delta(i, j) * gamma[j];
			}
		}

		// calc Sum(dxhat), Sum(dxhat * xhat)
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				sum1[j] += dxhat(i, j);
				sum2[j] += dxhat(i, j) * xhat(i, j);
			}
		}

		// calc dx, dgamma, dbeta
		float m = (float)batch;
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < n_feat; j++) {
				prev_delta(i, j) = (m * dxhat(i, j)) - sum1[j] - (xhat(i, j) * sum2[j]);
				prev_delta(i, j) /= m * std::sqrt(var[j] + eps);
				dgamma[j] += xhat(i, j) * this->delta(i, j);
				dbeta[j] += this->delta(i, j);
			}
		}
    #endif
	}

    template<typename T>
	void BatchNorm1d<T>::update_weight(float lr, float decay)
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
	void BatchNorm1d<T>::zero_grad()
	{
		this->delta.setZero();
		dxhat.setZero();
		dgamma.setZero();
		dbeta.setZero();
		sum1.setZero();
		sum2.setZero();
	}

    template<typename T>
	vector<int> BatchNorm1d<T>::output_shape() { return { batch, n_feat }; }
}

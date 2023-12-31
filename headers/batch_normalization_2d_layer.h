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
		VecX<T> move_mu;
		VecX<T> move_var;
		VecX<T> gamma;
		VecX<T> beta;
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

		move_mu.setZero();
		move_var.setZero();
		gamma.setConstant(1.f);
		beta.setZero();
	}

    template<typename T>
	void BatchNorm2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		if (is_training) {
			calc_batch_mu(prev_out);
			calc_batch_var(prev_out);
			normalize_and_shift(prev_out, is_training);
			// update moving mu and var
			move_mu = move_mu * momentum + mu * (1 - momentum);
			move_var = move_var * momentum + var * (1 - momentum);
		}
		else {
			normalize_and_shift(prev_out, is_training);
		}
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
		const float* M = mu.data();
		const float* V = var.data();

		if (!is_training) {
			M = move_mu.data();
			V = move_var.data();
		}

		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < ch; c++) {
				int i = c + ch * n;
				float m = M[c];
				float s = std::sqrt(V[c] + eps);
				float g = gamma[c];
				float b = beta[c];
				for (int j = 0; j < hw; j++) {
					xhat(i, j) = (prev_out(i, j) - m) / s;
					this->output(i, j) = g * xhat(i, j) + b;
				}
			}
		}
	}

    template<typename T>
	void BatchNorm2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
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
	}

    template<typename T>
	void BatchNorm2d<T>::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		if (t1 != 1) {
			gamma *= t1;
			beta *= t1;
		}
		gamma -= t2 * dgamma;
		beta -= t2 * dbeta;
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

#pragma once
#include "layer.h"

namespace simple_nn
{
	class Conv2d : public Layer
	{
	public:
		int batch;
		int in_chs;
		int out_chs;
		int in_h;
		int in_w;
		int ker_h;
		int ker_w;
		int out_h;
		int out_w;
		int pad;
		int out_block_size;
		int ker_block_size;
		string init_opt;
		float* kernel;
		float* dkernel;
		float* bias;
		float* dbias;
	public:
		Conv2d(int out_channels,
			int kernel_size,
			int pad,
			const vector<int>& input_shape,
			string init_opt = "normal");
		Conv2d(int out_channels,
			int kernel_size,
			int pad,
			string init_opt = "normal");
		~Conv2d();
		void set_layer(int batch_size, const vector<int>& input_shape) override;
		void forward_propagate(const float* prev_out, bool isPrediction) override;
		void backward_propagate(const float* prev_out, float* prev_delta, bool isFirst) override;
		void update_weight(float lr, float decay) override;
		vector<int> input_shape() override;
		vector<int> output_shape() override;
		int get_out_block_size() override;
	};

	Conv2d::Conv2d(int out_channels,
		int kernel_size,
		int pad,
		const vector<int>& input_shape,
		string init_opt) :
		Layer(CONV2D),
		batch(0),
		in_chs(input_shape[2]),
		out_chs(out_channels),
		in_h(input_shape[0]),
		in_w(input_shape[1]),
		ker_h(kernel_size),
		ker_w(kernel_size),
		out_h(calc_outsize(in_h, ker_h, 1, pad)),
		out_w(calc_outsize(in_w, ker_w, 1, pad)),
		pad(pad),
		out_block_size(0),
		ker_block_size(0),
		init_opt(init_opt)
	{
		if (init_opt != "normal" && init_opt != "uniform") {
			throw logic_error("Conv2D::Conv2D(int, int, ...): Invalid init option.");
		}
	}

	Conv2d::Conv2d(int out_channels,
		int kernel_size,
		int pad,
		string init_opt) :
		Layer(CONV2D),
		batch(0),
		in_chs(0),
		out_chs(out_channels),
		in_h(0),
		in_w(0),
		ker_h(kernel_size),
		ker_w(kernel_size),
		out_h(0),
		out_w(0),
		pad(pad),
		out_block_size(0),
		ker_block_size(0),
		init_opt(init_opt)
	{
		if (init_opt != "normal" && init_opt != "uniform") {
			throw logic_error("Conv2D::Conv2D(int, int, ...): Invalid init option.");
		}
	}

	Conv2d::~Conv2d()
	{
		delete_memory(output);
		delete_memory(delta);
		delete_memory(kernel);
		delete_memory(dkernel);
		delete_memory(bias);
		delete_memory(dbias);
	}

	void Conv2d::set_layer(int batch, const vector<int>& input_shape)
	{
		this->batch = batch;
		in_h = input_shape[0];
		in_w = input_shape[1];
		in_chs = input_shape[2];
		out_h = calc_outsize(in_h, ker_h, 1, pad);
		out_w = calc_outsize(in_w, ker_w, 1, pad);
		out_block_size = batch * out_chs * out_h * out_w;
		ker_block_size = out_chs * in_chs * ker_h * ker_w;

		allocate_memory(output, out_block_size);
		allocate_memory(delta, out_block_size);
		allocate_memory(kernel, ker_block_size);
		allocate_memory(dkernel, ker_block_size);
		allocate_memory(bias, out_chs);
		allocate_memory(dbias, out_chs);

		init_weight(kernel, ker_block_size, in_h * in_w * in_chs, out_h * out_w, init_opt);
		set_zero(dkernel, ker_block_size);
	}

	void im2col(const float* im, int channels,
		int in_h, int in_w,
		int out_h, int out_w,
		int ksize, int pad, float* im_col)
	{
		int im_col_height = ksize * ksize * channels;
		int im_col_width = out_h * out_w;

		int c, h, w;
		for (c = 0; c < im_col_height; c++) {
			int w_offset = c % ksize;
			int h_offset = (c / ksize) % ksize;
			int im_c = c / ksize / ksize;			// channel index
			for (h = 0; h < out_h; h++) {
				for (w = 0; w < out_w; w++) {
					int i = h + h_offset - pad;
					int j = w + w_offset - pad;
					int idx = (c * out_h + h) * out_w + w;
					if (i >= 0 && i < in_h &&
						j >= 0 && j < in_w) {
						im_col[idx] = im[j + in_w * (i + in_h * im_c)];
					}
					else {
						im_col[idx] = 0.0F;
					}
				}
			}
		}
	}

	void add_bias(float* C, int N, float bias)
	{
		std::for_each(C, C + N, [&](float& elem) { elem += bias; });
	}

	void Conv2d::forward_propagate(const float* prev_out, bool isPrediction)
	{
		/*
		* A: M x K = 1 x (ker_h * ker_w * in_channels)
		* B: K x N = (ker_h * ker_w * in_channels) x (out_h * out_w)
		*/
		int n, c;
		int M = 1;
		int N = out_h * out_w;
		int K = ker_h * ker_w * in_chs;
		float* space = new float[K * N];

		for (n = 0; n < batch; n++) {
			float* B = space;
			const float* im = prev_out + in_h * in_w * in_chs * n;
			im2col(im, in_chs, in_h, in_w, out_h, out_w, ker_h, pad, B);
			for (c = 0; c < out_chs; c++) {
				const float* A = kernel + c * K;
				float* C = output + N * (c + out_chs * n);
				gemm_nn(M, N, K, 1.0F, A, K, B, N, C, N);
				//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0F, A, K, B, N, 0.0F, C, N);
				add_bias(C, N, bias[c]);
			}
			set_zero(B, K * N);
		}
		delete_memory(space);
	}

	void Conv2d::backward_propagate(const float* prev_out, float* prev_delta, bool isFirst)
	{
		/*
		* A: M x K = 1 x (out_h x out_w)
		* B: K x N = (out_h * out_w) x ( ker_h * ker_w) 
		*/
		int n, c1, c2;
		int M = 1;
		int N = ker_h * ker_w;
		int K = out_h * out_w;
		float* space = new float[K * N];

		for (n = 0; n < batch; n++) {
			for (c1 = 0; c1 < in_chs; c1++) {
				float* B = space;
				const float* im = prev_out + in_h * in_w * (c1 + in_chs * n);
				im2col(im, 1, in_h, in_w, ker_h, ker_w, out_h, pad, B);
				for (c2 = 0; c2 < out_chs; c2++) {
					const float* A = delta + K * (c2 + out_chs * n);
					float* C = dkernel + N * (c1 + in_chs * c2);
					gemm_nn(M, N, K, 1.0F, A, K, B, N, C, N);
					//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0F, A, K, B, N, 0.0F, C, N);
				}
				set_zero(B, K * N);
			}
		}

		for (n = 0; n < batch; n++) {
			for (c1 = 0; c1 < out_chs; c1++) {
				float* A = delta + K * (c1 + out_chs * n);
				dbias[c1] += std::accumulate(A, A + K, 0.0F);
			}
		}

		delete_memory(space);

		if (!isFirst) {
			M = 1;
			N = in_h * in_w;
			K = ker_h * ker_w;
			float* space1 = new float[K * N];
			float* space2 = new float[K];

			int pad = ker_h - 1;
			for (n = 0; n < batch; n++) {
				for (c1 = 0; c1 < out_chs; c1++) {
					float* B = space1;
					const float* im = delta + out_h * out_w * (c1 + out_chs * n);
					im2col(im, 1, out_h, out_w, in_h, in_w, ker_h, pad, B);
					for (c2 = 0; c2 < in_chs; c2++) {
						float* A = space2;
						float* tmp = kernel + K * (c2 + in_chs * c1);
						std::reverse_copy(tmp, tmp + K, A);
						float* C = prev_delta + N * (c2 + in_chs * n);
						gemm_nn(M, N, K, 1.0F, A, K, B, N, C, N);
						//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0F, A, K, B, N, 0.0F, C, N);
					}
					set_zero(B, K * N);
				}
			}
			delete_memory(space1);
			delete_memory(space2);
		}
	}

	void Conv2d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;
		for (int i = 0; i < ker_block_size; i++) {
			kernel[i] = t1 * kernel[i] - t2 * dkernel[i];
			dkernel[i] = 0.0F;
		}
		for (int i = 0; i < out_chs; i++) {
			bias[i] = t1 * bias[i] - t2 * dbias[i];
			dbias[i] = 0.0F;
		}
	}

	vector<int> Conv2d::input_shape()
	{
		if (in_h == 0 || in_w == 0 || in_chs == 0) {
			throw logic_error("Conv2D::input_shape(): Input shape is empty.");
		}
		return { in_h, in_w, in_chs };
	}

	vector<int> Conv2d::output_shape() { return { out_h, out_w, out_chs }; }

	int Conv2d::get_out_block_size() { return out_block_size; }
}
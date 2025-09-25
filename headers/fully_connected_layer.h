#pragma once
#include "common.h"
#include "layer.h"

namespace simple_nn
{
    template<typename T>
	class Linear : public Layer<T>
	{
	private:
		int batch;
		int in_feat;
		int out_feat;
		string option;
		MatX<T> dW;
		RowVecX<T> db;
	public:
#if PUBLIC_WEIGHTS == 1
		MatX<UINT_TYPE> W;
		RowVecX<UINT_TYPE> b;
#else
		MatX<T> W;
		RowVecX<T> b;
#endif
		Linear(int in_features, int out_features, string option = "kaiming_uniform");
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Linear<T>::Linear(int in_features, int out_features, string option) :
		Layer<T>(LayerType::LINEAR),
		batch(0),
		in_feat(in_features),
		out_feat(out_features),
		option(option) {}

    template<typename T>
	void Linear<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];

		this->output.resize(batch, out_feat);
		this->delta.resize(batch, out_feat);
		W.resize(out_feat, in_feat);
		dW.resize(out_feat, in_feat);
		b.resize(out_feat);
		db.resize(out_feat);
        #if IS_TRAINING == 1
		init_weight(W, in_feat, out_feat, option);
		b.setZero();
        #endif
	}


    template<typename T>
	void Linear<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
#if PROTOCOL == 4 && FC_TRIPLES == 1 && PUBLIC_WEIGHTS == 0
        T::SetupFullyConnectedTriples(prev_out.data(), W.data(), this->output.data(), batch, in_feat, out_feat);
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
        
        for (int n = 0; n < batch; n++) {

#if PUBLIC_WEIGHTS == 1
            DATATYPE* W = this->W.data();
#else
            const T* W = this->W.data();
#endif
            const T* A = prev_out.data() + n * prev_out.cols(); 
            T* C = this->output.data() + n * this->output.cols();
            prepare_Matrix_Vector_Product(W, A, C, this->W.rows(), this->W.cols());
        }

            T::communicate();
            auto C = this->output.data();
            complete_GEMM(C, this->output.size());
            #if TRUNC_DELAYED == 0 && (TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4)
                trunc_2k_in_place(this->output.data(), this->output.size(),false);
            #elif TRUNC_DELAYED == 0 && TRUNC_APPROACH == 2
                trunc_exact_in_place(this->output.data(), this->output.size());
            #elif TRUNC_DELAYED == 0 && TRUNC_APPROACH == 3
                trunc_exact_opt_in_place(this->output.data(), this->output.size());
            #endif
          
            auto B = b.data();
            for (int n = 0; n < batch; n++) 
                for(int i = 0; i < this->output.cols(); ++i)
                    add_bias(C[n * this->output.cols() + i], B[i]);

}
	

    template<typename T>
	void Linear<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		// dW = delta(Vector) * prev_out(RowVector)
		// db = delta
		/* for (int n = 0; n < batch; n++) { */
		/* 	dW.noalias() += this->delta.row(n).transpose() * prev_out.row(n); */
		/* 	db.noalias() += this->delta.row(n); */
		/* } */

		/* // prev_delta = W.T * delta(Vector) */
		/* if (!this->is_first) { */
		/* 	for (int n = 0; n < batch; n++) { */
		/* 		prev_delta.row(n).noalias() = W.transpose() * this->delta.row(n).transpose(); */
		/* 	} */
		/* } */
	}

    template<typename T>
	void Linear<T>::update_weight(float lr, float decay)
	{
		/* float t1 = (1 - (2 * lr * decay) / batch); */
		/* float t2 = lr / batch; */

		/* if (t1 != 1) { */
		/* 	W *= t1; */
		/* 	b *= t1; */
		/* } */

		/* W -= t2 * dW; */
		/* b -= t2 * db; */
	}

    template<typename T>
	void Linear<T>::zero_grad()
	{
		this->delta.setZero();
		dW.setZero();
		db.setZero();
	}

    template<typename T>
	vector<int> Linear<T>::output_shape() { return { batch, out_feat }; }
}

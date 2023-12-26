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
		Linear(int in_features, int out_features, string option);
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
    /* if(current_phase == 1) */
    /* std::cout << "FC ..." << std::endl; */
        for (int n = 0; n < batch; n++) {
            for(int i = 0; i < W.rows(); ++i) {
            T sum = T(0);
            for(int j = 0; j < W.cols(); ++j) {

#if PUBLIC_WEIGHTS == 0
                sum += W(i, j).prepare_dot(prev_out(n, j));  // Use custom * and + operators
#else
                sum += prev_out(n, j).mult_public(W(i, j));  
#endif
            }


#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
            sum.mask_and_send_dot_without_trunc(); // send immediately to utilize network better
#else
            sum.mask_and_send_dot();
#endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    #else
            sum.prepare_mult_public_fixed(1); //initiate truncation
    #endif
#endif


            this->output(n, i) = sum;
        }
            /* tmp_output2.row(n).noalias() = W * prev_out.row(n).transpose(); */
        }

            T::communicate();
            /* for (int i = 0; i < this->output.size(); i++) */ 
            /*     this->output(i).mask_and_send_dot(); */
            for (int i = 0; i < this->output.size(); i++) {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                this->output(i).complete_mult_without_trunc();
#else
                this->output(i).complete_mult();
#endif
#else
    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
    #else
                this->output(i).complete_public_mult_fixed();
    #endif
#endif
                /* this->output(i) += b(i); //added to replace lower code */
            }

		    /* for (int n = 0; n < batch; n++) */ 
			    /* this->output.row(n).noalias() += b; */
            for (int n = 0; n < batch; n++) 
                for(int i = 0; i < this->output.cols(); ++i)
                    this->output(n, i) += b(i);
            


#if TRUNC_APPROACH == 1 && TRUNC_DELAYED == 0
            trunc_2k_in_place(this->output.data(), this->output.size());
#endif


		
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

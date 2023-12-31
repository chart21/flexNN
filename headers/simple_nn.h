#pragma once
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "average_pooling_layer.h"
#include "adaptive_average_pooling_layer.h"
#include "activation_layer.h"
#include "batch_normalization_1d_layer.h"
#include "batch_normalization_2d_layer.h"
#include "flatten_layer.h"
#include "loss_layer.h"
#include "optimizers.h"
#include "data_loader.h"
#include "file_manage.h"

namespace simple_nn
{
    template<typename T>
	class SimpleNN
	{
	protected:
		vector<Layer<T>*> net;
		Optimizer* optim;
		Loss<T>* loss;
	public:
		void add(Layer<T>* layer);
		virtual void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr);
		void fit(const DataLoader<T>& train_loader, int epochs, const DataLoader<T>& valid_loader);
		void save(string save_dir, string fname);
        template<int id>
		void load(string save_dir, string fname);
        template<typename F>
		void evaluate(const DataLoader<F>& data_loader);
	private:
		virtual void forward(const MatX<T>& X, bool is_training);
		void classify(const MatX<T>& output, VecXi& classified);
		/* void error_criterion(const VecXi& classified, const VecXi& labels, T& error_acc); */
		void error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc);
		void loss_criterion(const MatX<T>& output, const VecXi& labels, T& loss_acc);
		void zero_grad();
		void backward(const MatX<T>& X);
		void update_weight();
		int count_params();
        template<int id>
        void prepare_read_params(fstream& fs);
        template<int id>
        void complete_read_params();
		void write_or_read_params(fstream& fs, string mode);
	};

    template<typename T>
	void SimpleNN<T>::add(Layer<T>* layer) { net.push_back(layer); }

    template<typename T>
	void SimpleNN<T>::compile(vector<int> input_shape, Optimizer* optim, Loss<T>* loss)
	{
		// set optimizer & loss
		this->optim = optim;
		this->loss = loss;

		// set first & last layer
		net.front()->is_first = true;
		net.back()->is_last = true;

		// set network
		for (int l = 0; l < net.size(); l++) {
			if (l == 0) net[l]->set_layer(input_shape);
			else net[l]->set_layer(net[l - 1]->output_shape());
		}

		// set Loss layer
		if (loss != nullptr) {
			loss->set_layer(net.back()->output_shape());
		}
	}

    template<typename T>
	void SimpleNN<T>::fit(const DataLoader<T>& train_loader, int epochs, const DataLoader<T>& valid_loader)
	{
		if (optim == nullptr || loss == nullptr) {
			cout << "The model must be compiled before fitting the data." << endl;
			exit(1);
		}

		int batch = train_loader.input_shape()[0];
		int n_batch = train_loader.size();

		MatX<T> X;
		VecXi Y;
		VecXi classified(batch);

		for (int e = 0; e < epochs; e++) {
			T loss(0);
			/* T error(0); */
            float error(0);

			system_clock::time_point start = system_clock::now();
			for (int n = 0; n < n_batch; n++) {
				X = train_loader.get_x(n);
				Y = train_loader.get_y(n);

				forward(X, true);
				classify(net.back()->output, classified);
				error_criterion(classified, Y, error);

				zero_grad();
				loss_criterion(net.back()->output, Y, loss);
				backward(X);
				update_weight();

				cout << "[Epoch:" << setw(3) << e + 1 << "/" << epochs << ", ";
				cout << "Batch: " << setw(4) << n + 1 << "/" << n_batch << "]";

				if (n + 1 < n_batch) {
					cout << "\r";
				}
			}
			system_clock::time_point end = system_clock::now();
			duration<float> sec = end - start;

			T loss_valid(0); 
			/* T error_valid(0); */
			float error_valid(0);

			int n_batch_valid = valid_loader.size();
			if (n_batch_valid != 0) {
				for (int n = 0; n < n_batch_valid; n++) {
					X = valid_loader.get_x(n);
					Y = valid_loader.get_y(n);

					forward(X, false);
					classify(net.back()->output, classified);
					error_criterion(classified, Y, error_valid);
					loss_criterion(net.back()->output, Y, loss_valid);
				}
			}

			cout << fixed << setprecision(2);
			/* cout << " - t: " << sec.count() << 's'; */
			/* cout << " - loss: " << loss.reveal() / n_batch; */
			/* cout << " - error: " << (error / n_batch).reveal() * 100 << "%"; */
			/* cout << " - error: " << (error / n_batch) * 100 << "%"; */
			if (n_batch_valid != 0) {
				/* cout << " - loss(valid): " << loss_valid.reveal() / n_batch_valid; */
				/* cout << " - error(valid): " << (error_valid / n_batch_valid).reveal() * 100 << "%"; */
				/* cout << " - error(valid): " << (error_valid / n_batch_valid) * 100 << "%"; */
			}
			cout << endl;
		}
	}

    template<typename T>
	void SimpleNN<T>::forward(const MatX<T>& X, bool is_training)
	{
		for (int l = 0; l < net.size(); l++) {
			if (l == 0) net[l]->forward(X, is_training);
			else 
            {
                start_timer();
                net[l]->forward(net[l - 1]->output, is_training);
                stop_timer(toString(net[l]->type));
/* #if IS_TRAINING == 0 */
/*                 if (l > 1) */
/*                 { */
/*                     net.erase(net.begin() + l - 1); */
/*                     l--; */
/*                 } */
/* #endif */
		    }
		}
	}

    template<typename T>
	void SimpleNN<T>::classify(const MatX<T>& output, VecXi& classified)
	{
		// assume that the last layer is linear, not 2d.
        /* std::cout << "output:" << output.rows() << " " << output.cols() << "\n"; */
        /* std::cout << "classified:" << classified.size() << "\n"; */
        assert(output.rows()*(BASE_DIV) == classified.size()); // Adjusted because of sint
        //loop over all elements in output and save them in float Matrix
        
        for (int i = 0; i < output.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {
                output(i,j).prepare_reveal_to_all();
            }
        }
        T::communicate();
#if JIT_VEC == 1
        MatXf output_float(output.rows()*(BASE_DIV), output.cols()); // 32x10
        for (int i = 0; i < output.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {
                alignas(sizeof(DATTYPE)) UINT_TYPE tmp[BASE_DIV];
                output(i,j).complete_reveal_to_all(tmp);
                for (int k = 0; k < BASE_DIV; k++) {
                    output_float(i*(BASE_DIV)+k,j) = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(tmp[k]);
                }
        }
        } 
#else
        MatXf output_float(output.rows(), output.cols());

        for (int i = 0; i < output.rows(); i++) {
            for (int j = 0; j < output.cols(); j++) {

                output_float(i,j) = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output(i,j).complete_reveal_to_all_single());
                /* output_float(i,j) = 0; */
            }
        }
#endif
        
        for (int i = 0; i < classified.size(); i++) {
			/* output.row(i).maxCoeff(&classified[i]); */
			output_float.row(i).maxCoeff(&classified[i]);
		}
    }

    template<typename T>
	void SimpleNN<T>::error_criterion(const VecXi& classified, const VecXi& labels, float& error_acc)
	/* void SimpleNN<T>::error_criterion(const VecXi& classified, const VecXi& labels, T& error_acc) */
	{
		int batch = (int)classified.size();
        /* assert(labels.size() >= batch); */

		/* T error(0); */
        float error(0);
		for (int i = 0; i < batch; i++) {
			if (classified[i] != labels[i]) 
            {
                error+=1;
                /* error+=T(1); */
                /* std::cout << "classified[i] = " << classified[i] << " labels[i] = " << labels[i] << "\n"; */
		    }
        }
		/* error_acc += error / batch; */
        error_acc += error;
	}

    template<typename T>
	void SimpleNN<T>::loss_criterion(const MatX<T>& output, const VecXi& labels, T& loss_acc)
	{
		loss_acc += loss->calc_loss(output, labels, net.back()->delta);
	}

    template<typename T>
	void SimpleNN<T>::zero_grad()
	{
		for (const auto& l : net) l->zero_grad();
	}

    template<typename T>
	void SimpleNN<T>::backward(const MatX<T>& X)
	{
		for (int l = (int)net.size() - 1; l >= 0; l--) {
			if (l == 0) {
				MatX<T> empty;
				net[l]->backward(X, empty);
			}
			else {
				net[l]->backward(net[l - 1]->output, net[l - 1]->delta);
			}
		}
	}

    template<typename T>
	void SimpleNN<T>::update_weight()
	{
		float lr = optim->lr();
		float decay = optim->decay();
		for (const auto& l : net) {
			l->update_weight(lr, decay);
		}
	}

    template<typename T>
	void SimpleNN<T>::save(string save_dir, string fname)
	{
		string path = save_dir + "/" + fname;
		fstream fout(path, ios::out | ios::binary);

		int total_params = count_params();
		fout.write((char*)&total_params, sizeof(int));

		write_or_read_params(fout, "write");
		cout << "Model parameters are saved in " << path << endl;

		fout.close();

		return;
	}
    
    template<typename T>
    template<int id>
	void SimpleNN<T>::load(string save_dir, string fname)
	{
		string path = save_dir + "/" + fname;
		fstream fin(path, ios::in | ios::binary);

		if (!fin) {
			cout << path << " does not exist. Setting dummy weights." << endl;
			/* exit(1); */
		}

		int total_params;
        if(!(!fin))
            fin.read((char*)&total_params, sizeof(int));
        else
            total_params = count_params();

		if (total_params != count_params()) {
			cout << "The number of parameters does not match." << endl;
			fin.close();
			exit(1);
		}
        prepare_read_params<id>(fin);
#if PUBLIC_WEIGHTS == 0
        T::communicate();
        complete_read_params<id>();
#endif
		/* write_or_read_params(fin, "read"); */

		fin.close();

		return;
	}

    template<typename T>
	int SimpleNN<T>::count_params()
	{
		int total_params = 0;
		for (const Layer<T>* l : net) {
			if (l->type == LayerType::LINEAR) {
				const Linear<T>* lc = dynamic_cast<const Linear<T>*>(l);
				total_params += (int)lc->W.size();
				total_params += (int)lc->b.size();
			}
			else if (l->type == LayerType::CONV2D) {
				const Conv2d<T>* lc = dynamic_cast<const Conv2d<T>*>(l);
				total_params += (int)lc->kernel.size();
				total_params += (int)lc->bias.size();
			}
			else if (l->type == LayerType::BATCHNORM1D) {
				const BatchNorm1d<T>* lc = dynamic_cast<const BatchNorm1d<T>*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
			}
			else if (l->type == LayerType::BATCHNORM2D) {
				const BatchNorm2d<T>* lc = dynamic_cast<const BatchNorm2d<T>*>(l);
				total_params += (int)lc->move_mu.size();
				total_params += (int)lc->move_var.size();
				total_params += (int)lc->gamma.size();
				total_params += (int)lc->beta.size();
			}
			else {
				continue;
			}
		}
		return total_params;
	}

template <typename T>
template <int id>
void SimpleNN<T>::prepare_read_params(fstream& fs)
{
    for (Layer<T>* l : net) {
        vector<float> tempMatrix1, tempMatrix2, tempMatrix3, tempMatrix4; // Temporary vectors for parameter storage

        if (l->type == LayerType::LINEAR) {
            Linear<T>* lc = dynamic_cast<Linear<T>*>(l);
            int s1 = lc->W.rows() * lc->W.cols();
            int s2 = lc->b.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);

            /* if (mode == "write") { */
                /* for (int i = 0; i < s1; i++) */ 
                /* { */
                /*     tempMatrix1[i] = lc->W(i / lc->W.cols(), i % lc->W.cols()).reveal(); */
                /* } */
                /* for (int i = 0; i < s2; i++) */
                /* { */
                /*     tempMatrix2[i] = lc->b[i].reveal(); */
                /* } */
                /* fs.write((char*)tempMatrix1.data(), sizeof(float) * s1); */
                /* fs.write((char*)tempMatrix2.data(), sizeof(float) * s2); */
            /* } */
            /* else { */
            if(!(!fs))
            {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
            }
                for (int i = 0; i < s1; i++) 
                {
#if PUBLIC_WEIGHTS == 0
                    lc->W(i / lc->W.cols(), i % lc->W.cols()).template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]));
#else
                    lc->W(i / lc->W.cols(), i % lc->W.cols()) = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]);
#endif
                }
                for (int i = 0; i < s2; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->b[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix2[i]));
#else
                    lc->b[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix2[i]);
#endif
                }
            }
        /* } */
        else if (l->type == LayerType::CONV2D) {
            Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l);
            int s1 = lc->kernel.rows() * lc->kernel.cols();
            int s2 = lc->bias.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);

            /* if (mode == "write") { */
                /* for (int i = 0; i < s1; i++) */ 
                /* { */
                /*     tempMatrix1[i] = lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).reveal(); */
                /* } */
                /* for (int i = 0; i < s2; i++) */ 
                /* { */
                /*     tempMatrix2[i] = lc->bias[i].reveal(); */
                /* } */
                /* fs.write((char*)tempMatrix1.data(), sizeof(float) * s1); */
                /* fs.write((char*)tempMatrix2.data(), sizeof(float) * s2); */
            /* } */
            /* else { */
            if(!(!fs))
            {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
            }

                for (int i = 0; i < s1; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]));
#else
                    lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()) = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]);
#endif
                } 
                for (int i = 0; i < s2; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->bias[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix2[i]));
#else
                    lc->bias[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix2[i]);
#endif
                }
            }

        else if (l->type == LayerType::BATCHNORM1D) {
            BatchNorm1d<T>* lc = dynamic_cast<BatchNorm1d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if(!(!fs))
            {
            fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
            fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
            fs.read((char*)tempMatrix3.data(), sizeof(float) * s3);
            fs.read((char*)tempMatrix4.data(), sizeof(float) * s4);
            }
                for (int i = 0; i < s1; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->move_mu[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]));
#else
                    lc->move_mu[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]);
#endif
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
#if PUBLIC_WEIGHTS == 0
                    lc->move_var[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(var));
#else
                    lc->move_var[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(var);
#endif
                }
                for (int i = 0; i < s3; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->gamma[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix3[i]));
#else
                    lc->gamma[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix3[i]);
#endif
                }
                for (int i = 0; i < s4; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->beta[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix4[i]));
#else
                    lc->beta[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix4[i]);
#endif
                }
        }
        else if (l->type == LayerType::BATCHNORM2D) {
            BatchNorm2d<T>* lc = dynamic_cast<BatchNorm2d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if(!(!fs))
            {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
                fs.read((char*)tempMatrix3.data(), sizeof(float) * s3);
                fs.read((char*)tempMatrix4.data(), sizeof(float) * s4);
            }
                for (int i = 0; i < s1; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->move_mu[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]));
#else
                    lc->move_mu[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix1[i]);
#endif
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
#if PUBLIC_WEIGHTS == 0
                    lc->move_var[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(var));
#else
                    lc->move_var[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(var);
#endif
                }
                for (int i = 0; i < s3; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->gamma[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix3[i]));
#else
                    lc->gamma[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix3[i]);
#endif
                }
                for (int i = 0; i < s4; i++)
                {
#if PUBLIC_WEIGHTS == 0
                    lc->beta[i].template prepare_receive_and_replicate<id>(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix4[i]));
#else
                    lc->beta[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(tempMatrix4[i]);
#endif
                }

        }
    }
    /* } */
}


    /* template<typename T> */
/* void SimpleNN<T>::prepare_read_params(fstream& fs) */
/* { */
    /* for (Layer<T>* l : net) { */

    /*     if (l->type == LayerType::LINEAR) { */
    /*         Linear<T>* lc = dynamic_cast<Linear<T>*>(l); */
    /*         int s1 = lc->W.rows() * lc->W.cols(); */
    /*         int s2 = lc->b.size(); */

    /*             for (int i = 0; i < s1; i++) */ 
    /*             { */
    /*                 lc->W(i / lc->W.cols(), i % lc->W.cols()).template prepare_receive_from<P_0>(); */
    /*             } */
    /*             for (int i = 0; i < s2; i++) */
    /*             { */
    /*                 lc->b[i].template prepare_receive_from<P_0>(); */
    /*             } */
    /*         } */
    /*     else if (l->type == LayerType::CONV2D) { */
    /*         Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l); */
    /*         int s1 = lc->kernel.rows() * lc->kernel.cols(); */
    /*         int s2 = lc->bias.size(); */

    /*             for (int i = 0; i < s1; i++) */
    /*             { */
    /*                 lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).template prepare_receive_from<P_0>(); */
    /*             } */ 
    /*             for (int i = 0; i < s2; i++) */
    /*             { */
    /*                 lc->bias[i].template prepare_receive_from<P_0>(); */
    /*             } */
    /*         } */
    /* } */
/* } */
    
template <typename T>
template <int id>
void SimpleNN<T>::complete_read_params()
{
    for (Layer<T>* l : net) {

        if (l->type == LayerType::LINEAR) {
            Linear<T>* lc = dynamic_cast<Linear<T>*>(l);
            int s1 = lc->W.rows() * lc->W.cols();
            int s2 = lc->b.size();

                for (int i = 0; i < s1; i++) 
                {
                    lc->W(i / lc->W.cols(), i % lc->W.cols()).template complete_receive_from<id>();
                }
                for (int i = 0; i < s2; i++)
                {
                    lc->b[i].template complete_receive_from<id>();
                }
            }
        else if (l->type == LayerType::CONV2D) {
            Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l);
            int s1 = lc->kernel.rows() * lc->kernel.cols();
            int s2 = lc->bias.size();

                for (int i = 0; i < s1; i++)
                {
                    lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).template complete_receive_from<id>();
                } 
                for (int i = 0; i < s2; i++)
                {
                    lc->bias[i].template complete_receive_from<id>();
                }
            }

        else if (l->type == LayerType::BATCHNORM1D)
        {
                BatchNorm1d<T>* lc = dynamic_cast<BatchNorm1d<T>*>(l);
				int s1 = (int)lc->move_mu.size();
				int s2 = (int)lc->move_var.size();
				int s3 = (int)lc->gamma.size();
				int s4 = (int)lc->beta.size();
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s2; i++)
                {
                    lc->move_var[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i].template complete_receive_from<id>();
                }
			}
        else if (l->type == LayerType::BATCHNORM2D)
        {
                BatchNorm2d<T>* lc = dynamic_cast<BatchNorm2d<T>*>(l);
				int s1 = (int)lc->move_mu.size();
				int s2 = (int)lc->move_var.size();
				int s3 = (int)lc->gamma.size();
				int s4 = (int)lc->beta.size();
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s2; i++)
                {
                    lc->move_var[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i].template complete_receive_from<id>();
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i].template complete_receive_from<id>();
                }

        }
    }
}


/* template<typename T> */
/* void SimpleNN<T>::write_or_read_params(fstream& fs, string mode) */
/* { */
/*     for (Layer<T>* l : net) { */
/*         vector<float> tempMatrix1, tempMatrix2, tempMatrix3, tempMatrix4; // Temporary vectors for parameter storage */

/*         if (l->type == LayerType::LINEAR) { */
/*             Linear<T>* lc = dynamic_cast<Linear<T>*>(l); */
/*             int s1 = lc->W.rows() * lc->W.cols(); */
/*             int s2 = lc->b.size(); */
/*             tempMatrix1.resize(s1); */
/*             tempMatrix2.resize(s2); */

/*             if (mode == "write") { */
/*                 for (int i = 0; i < s1; i++) */ 
/*                 { */
/*                     tempMatrix1[i] = lc->W(i / lc->W.cols(), i % lc->W.cols()).reveal(); */
/*                 } */
/*                 for (int i = 0; i < s2; i++) */
/*                 { */
/*                     tempMatrix2[i] = lc->b[i].reveal(); */
/*                 } */
/*                     fs.write((char*)tempMatrix1.data(), sizeof(float) * s1); */
/*                 fs.write((char*)tempMatrix2.data(), sizeof(float) * s2); */
/*             } */
/*             else { */
/*                 fs.read((char*)tempMatrix1.data(), sizeof(float) * s1); */
/*                 fs.read((char*)tempMatrix2.data(), sizeof(float) * s2); */
/*                 for (int i = 0; i < s1; i++) */ 
/*                 { */
/*                     lc->W(i / lc->W.cols(), i % lc->W.cols()) = T(tempMatrix1[i]); */
/*                 } */
/*                 for (int i = 0; i < s2; i++) */
/*                 { */
/*                     lc->b[i] = T(tempMatrix2[i]); */
/*                 } */
/*             } */
/*         } */
/*         else if (l->type == LayerType::CONV2D) { */
/*             Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l); */
/*             int s1 = lc->kernel.rows() * lc->kernel.cols(); */
/*             int s2 = lc->bias.size(); */
/*             tempMatrix1.resize(s1); */
/*             tempMatrix2.resize(s2); */

/*             if (mode == "write") { */
/*                 for (int i = 0; i < s1; i++) */ 
/*                 { */
/*                     tempMatrix1[i] = lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).reveal(); */
/*                 } */
/*                 for (int i = 0; i < s2; i++) */ 
/*                 { */
/*                     tempMatrix2[i] = lc->bias[i].reveal(); */
/*                 } */
/*                 fs.write((char*)tempMatrix1.data(), sizeof(float) * s1); */
/*                 fs.write((char*)tempMatrix2.data(), sizeof(float) * s2); */
/*             } */
/*             else { */
/*                 fs.read((char*)tempMatrix1.data(), sizeof(float) * s1); */
/*                 fs.read((char*)tempMatrix2.data(), sizeof(float) * s2); */
/*                 for (int i = 0; i < s1; i++) */
/*                 { */
/*                     lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()) = T(tempMatrix1[i]); */
/*                 } */ 
/*                 for (int i = 0; i < s2; i++) */
/*                 { */
/*                     lc->bias[i] = T(tempMatrix2[i]); */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */


    /* template<typename T> */
	/* void SimpleNN<T>::write_or_read_params(fstream& fs, string mode) */
	/* { */
		/* for (const Layer<T>* l : net) { */
			/* if (l->type == LayerType::LINEAR) { */
				/* const Linear<T>* lc = dynamic_cast<const Linear<T>*>(l); */
				/* int s1 = (int)lc->W.size(); */
				/* int s2 = (int)lc->b.size(); */
				/* if (mode == "write") { */
					/* fs.write((char*)lc->W.data(), sizeof(float) * s1); */
					/* fs.write((char*)lc->b.data(), sizeof(float) * s2); */
				/* } */
				/* else { */
					/* fs.read((char*)lc->W.data(), sizeof(float) * s1); */
					/* fs.read((char*)lc->b.data(), sizeof(float) * s2); */
				/* } */
			/* } */
			/* else if (l->type == LayerType::CONV2D) { */
				/* const Conv2d<T>* lc = dynamic_cast<const Conv2d<T>*>(l); */
				/* int s1 = (int)lc->kernel.size(); */
				/* int s2 = (int)lc->bias.size(); */
				/* if (mode == "write") { */
					/* fs.write((char*)lc->kernel.data(), sizeof(float) * s1); */
					/* fs.write((char*)lc->bias.data(), sizeof(float) * s2); */
				/* } */
				/* else { */
					/* fs.read((char*)lc->kernel.data(), sizeof(float) * s1); */
					/* fs.read((char*)lc->bias.data(), sizeof(float) * s2); */
				/* } */
			/* } */
			/* else if (l->type == LayerType::BATCHNORM1D) { */
				/* const BatchNorm1d<T>* lc = dynamic_cast<const BatchNorm1d<T>*>(l); */
				/* int s1 = (int)lc->move_mu.size(); */
				/* int s2 = (int)lc->move_var.size(); */
				/* int s3 = (int)lc->gamma.size(); */
				/* int s4 = (int)lc->beta.size(); */
				/* if (mode == "write") { */
					/* fs.write((char*)lc->move_mu.data(), sizeof(float) * s1); */
					/* fs.write((char*)lc->move_var.data(), sizeof(float) * s2); */
					/* fs.write((char*)lc->gamma.data(), sizeof(float) * s3); */
					/* fs.write((char*)lc->beta.data(), sizeof(float) * s4); */
				/* } */
				/* else { */
					/* fs.read((char*)lc->move_mu.data(), sizeof(float) * s1); */
					/* fs.read((char*)lc->move_var.data(), sizeof(float) * s2); */
					/* fs.read((char*)lc->gamma.data(), sizeof(float) * s3); */
					/* fs.read((char*)lc->beta.data(), sizeof(float) * s4); */
				/* } */
			/* } */
			/* else if (l->type == LayerType::BATCHNORM2D) { */
				/* const BatchNorm2d<T>* lc = dynamic_cast<const BatchNorm2d<T>*>(l); */
				/* int s1 = (int)lc->move_mu.size(); */
				/* int s2 = (int)lc->move_var.size(); */
				/* int s3 = (int)lc->gamma.size(); */
				/* int s4 = (int)lc->beta.size(); */
				/* if (mode == "write") { */
					/* fs.write((char*)lc->move_mu.data(), sizeof(float) * s1); */
					/* fs.write((char*)lc->move_var.data(), sizeof(float) * s2); */
					/* fs.write((char*)lc->gamma.data(), sizeof(float) * s3); */
					/* fs.write((char*)lc->beta.data(), sizeof(float) * s4); */
				/* } */
				/* else { */
					/* fs.read((char*)lc->move_mu.data(), sizeof(float) * s1); */
					/* fs.read((char*)lc->move_var.data(), sizeof(float) * s2); */
					/* fs.read((char*)lc->gamma.data(), sizeof(float) * s3); */
					/* fs.read((char*)lc->beta.data(), sizeof(float) * s4); */
				/* } */
			/* } */
			/* else { */
				/* continue; */
			/* } */
		/* } */
	/* } */

    template<typename T>
    template<typename F>
	void SimpleNN<T>::evaluate(const DataLoader<F>& data_loader)
	{
		int batch = data_loader.input_shape()[0];
        int ch = data_loader.input_shape()[1];
		int n_batch = data_loader.size();
        /* std::cout << "batch: " << batch << "\n"; */
        /* std::cout << "n_batch: " << n_batch << "\n"; */
		/* T error_acc(0); */
        float error_acc(0);

		MatX<T> X;
		VecXi Y;
		VecXi classified(batch); //Adjusted because of sint

		system_clock::time_point start = system_clock::now();
		for (int n = 0; n < n_batch; n++) {
			auto test_X = data_loader.get_x(n); //Adjusted because of sint
			VecXi Y = data_loader.get_y(n);
#if JIT_VEC == 1 
			/* MatX<float> test_X = data_loader.get_x(n); //Adjusted because of sint */
            MatX<T> test_XX(test_X.rows()/(BASE_DIV), test_X.cols());
    for (int j = 0; j < test_X.cols(); j++) {
        for (int i = 0; i < test_X.rows(); i+=BASE_DIV*ch) {
            if(i+BASE_DIV*ch > test_X.rows()) {
                break; // do not process leftovers
            }
        alignas(sizeof(DATATYPE)) UINT_TYPE tmp[ch][BASE_DIV];
#if BASETYPE == 1
        alignas(sizeof(DATATYPE)) DATATYPE tmp2[ch][BITLENGTH];
#else
        alignas(sizeof(DATATYPE)) DATATYPE tmp2[ch];
#endif
        for( int c = 0; c < ch; c++)
            for (int k = 0; k < BASE_DIV; ++k) {
                tmp[c][k] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(test_X(i+k*ch+c, j));
            }
        for( int c = 0; c < ch; c++)
        {
#if BASETYPE == 1
        orthogonalize_arithmetic(tmp[c], tmp2[c]);
#else
        orthogonalize_arithmetic(tmp[c],&tmp2[c],1);
#endif
            test_XX(i / (BASE_DIV) + c, j).template prepare_receive_from<DATAOWNER>(tmp2[c]);
        }
    }
}
    T::communicate();
    for (int j = 0; j < test_XX.cols(); ++j) {
        for (int i = 0; i < test_XX.rows(); ++i) {
            test_XX(i, j).template complete_receive_from<DATAOWNER>();
        }
    }
			forward(test_XX, false);
#else
			forward(test_X, false);
#endif
			classify(net.back()->output, classified);
			error_criterion(classified, Y, error_acc);
		if(current_phase == 1)	
        {
			cout << "[Batch: " << setw(3) << n + 1 << "/" << n_batch << "]";
			if (n + 1 < n_batch) {
				cout << "\r";
			}
            
		}
        }
		system_clock::time_point end = system_clock::now();
		duration<float> sec = end - start;

	    if(current_phase == 1)	
        {
        cout << fixed << setprecision(2);
		cout << " - t: " << sec.count() << "s";
		cout << " - error(" << batch * n_batch << " images): ";

		/* cout << error_acc.reveal() / (batch * n_batch * DATTYPE) * 100 << "%" << endl; */
		cout << error_acc / (batch * n_batch) * 100 << "%" << endl;
        }
	}
}

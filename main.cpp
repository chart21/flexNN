#include "headers/simple_nn.h"
#include "headers/config.h"
using namespace std;
using namespace simple_nn;
using namespace Eigen;

template<typename T>
void load_model(const Config& cfg, SimpleNN<T>& model);

int main(int argc, char** argv)
{
    using DATATYPE = uint16_t;
    using FLOATTYPE = float;
    using UINTTYPE = uint16_t;
    using INTTYPE = int16_t;
    using SHARETYPE = Wrapper<FLOATTYPE, INTTYPE, UINTTYPE, FRACTIONAL_VALUE, DATATYPE>;

    /* using Sharetype = Wrapper<DATATYPE>; */
    /* using F = FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE> ; */
	Config cfg;
	cfg.parse(argc, argv);
	cfg.print_config();

	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28;

	MatX<float> train_X, test_X;

	VecXi train_Y, test_Y;

	DataLoader<SHARETYPE> train_loader, test_loader;

	if (cfg.mode == "train") {
		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train);
		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train);
        MatX<SHARETYPE> train_XX = train_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
});

		train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train);
	}

    std::cout << "Reading MNIST test data..." << std::endl;
	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test);
	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test);
    
    MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { 
    return SHARETYPE(val);
    });
	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test);

	cout << "Dataset loaded." << endl;

    SimpleNN<SHARETYPE> model;
	load_model(cfg, model);
	/* model.add(new Conv2d<SHARETYPE>(1, 6, 5, 2, cfg.init)); */
    /* model.add(new ReLU<SHARETYPE>); */
    /* model.add(new MaxPool2d<SHARETYPE>(2, 2)); */
    /* model.add(new Flatten<SHARETYPE>); */
    /* model.add(new Linear<SHARETYPE>(400, 120, cfg.init)); */
	/* model.add(new Softmax<SHARETYPE>); */

    /* model.add(new AvgPool2d<SHARETYPE>(2, 2)); */
    /* model.add(new BatchNorm2d<SHARETYPE>); */

	cout << "Model construction completed." << endl;

    if (cfg.mode == "train") {
        if (cfg.loss == "cross_entropy") {
            model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss<SHARETYPE>);
        }
        else {
            model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss<SHARETYPE>);
        }
        model.fit(train_loader, cfg.epoch, test_loader);
        model.save("./model_zoo", cfg.model + ".pth");
    }
    
    else {
        model.compile({ cfg.batch, ch, h, w });
        std::cout << "Loading Model Parameters..." << std::endl;
        model.load(cfg.save_dir, cfg.pretrained);
        model.evaluate(test_loader);
    }

	return 0;
}



template<typename T, typename S>
void load_model(const Config& cfg, SimpleNN<T>& model)
{
	if (cfg.model == "lenet5") {
		for (int i = 0; i < 6; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Conv2d<T>(1, 6, 5, 2, cfg.init));
				}
				else {
					model.add(new Conv2d<T>(6, 16, 5, 0, cfg.init));
				}
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm2d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
				if (cfg.pool == "max") {
					model.add(new MaxPool2d<T>(2, 2));
				}
				else {
					model.add(new AvgPool2d<T>(2, 2));
				}
			}
			else if (i == 2) {
				model.add(new Flatten<T>);
			}
			else if (i < 5) {
				if (i == 3) {
					model.add(new Linear<T>(400, 120, cfg.init));
				}
				else {
					model.add(new Linear<T>(120, 84, cfg.init));
				}
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(84, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax<S>);
				}
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			if (i < 2) {
				if (i == 0) {
					model.add(new Linear<T>(784, 500, cfg.init));
				}
				else {
					model.add(new Linear<T>(500, 150, cfg.init));
				}
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.activ == "relu") {
					model.add(new ReLU<T>);
				}
				/* else { */
				/* 	model.add(new Tanh<T>); */
				/* } */
			}
			else {
				model.add(new Linear<T>(150, 10, cfg.init));
				/* if (cfg.use_batchnorm) { */
				/* 	model.add(new BatchNorm1d<T>); */
				/* } */
				if (cfg.loss == "cross_entropy") {
					model.add(new Softmax<S>);
				}
				/* else { */
				/* 	model.add(new Sigmoid<T>); */
				/* } */
			}
		}
	}
}

/* int main_old(int argc, char** argv) */
/* { */
/*     using DATATYPE = float; */
/*     using FLOATTYPE = float; */
/*     using UINTTYPE = float; */
/*     using SHARETYPE = Share<DATATYPE>; */
/* 	Config cfg; */
/* 	cfg.parse(argc, argv); */
/* 	cfg.print_config(); */

/* 	int n_train = 60000, n_test = 10000, ch = 1, h = 28, w = 28; */

/* 	MatX<float> train_X, test_X; */

/* 	VecXi train_Y, test_Y; */

/* 	DataLoader<SHARETYPE> train_loader, test_loader; */

/* 	if (cfg.mode == "train") { */
/* 		train_X = read_mnist(cfg.data_dir, "train-images.idx3-ubyte", n_train); */
/* 		train_Y = read_mnist_label(cfg.data_dir, "train-labels.idx1-ubyte", n_train); */
/*         MatX<SHARETYPE> train_XX = train_X.unaryExpr([](float val) { */ 
/*     return SHARETYPE( FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE>::fixedToFloat(val)); */ 
/* }); */

/* 		train_loader.load(train_XX, train_Y, cfg.batch, ch, h, w, cfg.shuffle_train); */
/* 	} */

/* 	test_X = read_mnist(cfg.data_dir, "t10k-images.idx3-ubyte", n_test); */
/* 	test_Y = read_mnist_label(cfg.data_dir, "t10k-labels.idx1-ubyte", n_test); */
/*     MatX<SHARETYPE> test_XX = test_X.unaryExpr([](float val) { */ 
/*     return SHARETYPE( FloatFixedConverter<FLOATTYPE, UINTTYPE, ANOTHER_FRACTIONAL_VALUE>::fixedToFloat(val)); */
/*     }); */
/* 	test_loader.load(test_XX, test_Y, cfg.batch, ch, h, w, cfg.shuffle_test); */

/* 	cout << "Dataset loaded." << endl; */

/* 	SimpleNN<SHARETYPE> model; */
/* 	load_model(cfg, model); */

/* 	cout << "Model construction completed." << endl; */

/* 	if (cfg.mode == "train") { */
/* 		if (cfg.loss == "cross_entropy") { */
/* 			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new CrossEntropyLoss<SHARETYPE>); */
/* 		} */
/* 		else { */
/* 			model.compile({ cfg.batch, ch, h, w }, new SGD(cfg.lr, cfg.decay), new MSELoss<SHARETYPE>); */
/* 		} */
/* 		model.fit(train_loader, cfg.epoch, test_loader); */
/* 		model.save("./model_zoo", cfg.model + ".pth"); */
/* 	} */
/* 	else { */
/* 		model.compile({ cfg.batch, ch, h, w }); */
/* 		model.load(cfg.save_dir, cfg.pretrained); */
/* 		model.evaluate(test_loader); */
/* 	} */

/* 	return 0; */
/* } */


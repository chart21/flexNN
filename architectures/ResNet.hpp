#pragma once
#include "../headers/simple_nn.h"
#include <queue>

// Conv2d(int in_channels, int out_channels, int kernel_size, int padding, string option);
// Linear(int in_features, int out_features, string option);
// MaxPool2d(int kernel_size, int stride);
// AvgPool2d(int kernel_size, int stride);

using namespace simple_nn;
/*
 class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
*/



/* class ResNet(nn.Module): */
/*     def __init__(self, block, layers, image_channels, num_classes): */
/*         super(ResNet, self).__init__() */
/*         self.in_channels = 64 */
/*         self.conv1 = nn.Conv2d( */
/*             image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False */
/*         ) */
/*         self.bn1 = nn.BatchNorm2d(64) */
/*         self.relu = nn.ReLU() */
/*         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) */

/*         # Essentially the entire ResNet architecture are in these 4 lines below */
/*         self.layer1 = self._make_layer( */
/*             block, layers[0], intermediate_channels=64, stride=1 */
/*         ) */
/*         self.layer2 = self._make_layer( */
/*             block, layers[1], intermediate_channels=128, stride=2 */
/*         ) */
/*         self.layer3 = self._make_layer( */
/*             block, layers[2], intermediate_channels=256, stride=2 */
/*         ) */
/*         self.layer4 = self._make_layer( */
/*             block, layers[3], intermediate_channels=512, stride=2 */
/*         ) */

/*         self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) */
/*         self.fc = nn.Linear(512 * 4, num_classes) */

/*     def forward(self, x): */
/*         x = self.conv1(x) */
/*         x = self.bn1(x) */
/*         x = self.relu(x) */
/*         x = self.maxpool(x) */
/*         x = self.layer1(x) */
/*         x = self.layer2(x) */
/*         x = self.layer3(x) */
/*         x = self.layer4(x) */

/*         x = self.avgpool(x) */
/*         x = x.reshape(x.shape[0], -1) */
/*         x = self.fc(x) */

/*         return x */

/* def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride): */
/*         identity_downsample = None */
/*         layers = [] */

/*         # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes */
/*         # we need to adapt the Identity (skip connection) so it will be able to be added */
/*         # to the layer that's ahead */
/*         if stride != 1 or self.in_channels != intermediate_channels * 4: */
/*             identity_downsample = nn.Sequential( */
/*                 nn.Conv2d( */
/*                     self.in_channels, */
/*                     intermediate_channels * 4, */
/*                     kernel_size=1, */
/*                     stride=stride, */
/*                     bias=False, */
/*                 ), */
/*                 nn.BatchNorm2d(intermediate_channels * 4), */
/*             ) */

/*         layers.append( */
/*             block(self.in_channels, intermediate_channels, identity_downsample, stride) */
/*         ) */

/*         # The expansion size is always 4 for ResNet 50,101,152 */
/*         self.in_channels = intermediate_channels * 4 */

/*         # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer, */
/*         # then finally back to 256. Hence no identity downsample is needed, since stride = 1, */
/*         # and also same amount of channels. */
/*         for i in range(num_residual_blocks - 1): */
/*             layers.append(block(self.in_channels, intermediate_channels)) */

/*         return nn.Sequential(*layers) */
template <typename T>
class ResNet : public SimpleNN<T>
{
private:
    int in_channels;
    vector<int> identity_layers;
    vector<string> identity_layers_type;
public:
    ResNet(int residual_blocks[4], int image_channels, int num_classes, string option = "kaiming_uniform") {
        in_channels = 64;
        this->add(new Conv2d<T>(image_channels, 64, 7, 2, 3, false, option));
        this->add(new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new MaxPool2d<T>(3, 2, 1));
        /* this->add(new AvgPool2d<T>(3, 2)); */
        /* this->add( new AvgPool2d<T>(3, 2)); */
        this->make_layer( residual_blocks[0], 64, 1, option);
        this->make_layer( residual_blocks[1], 128, 2, option);
        this->make_layer( residual_blocks[2], 256, 2, option);
        this->make_layer( residual_blocks[3], 512, 2, option);
        /* this->add( new AvgPool2d<T>(1, 1)); */
        this->add( new AdaptiveAvgPool2d<T>(1, 1));
        this->add( new Flatten<T>());
        this->add(new Linear<T>(512 * 4, num_classes, option));
    }

    void add_identity_layer(string type) {
        this->identity_layers.push_back(this->net.size());
        this->identity_layers_type.push_back(type);
    }
 
    void add_block(int in_channels, int intermediate_channels, bool identity_downsample, int stride, string option) {
        const int expansion = 4;
        this->add_identity_layer("Identity_Store");
        this->add( new Conv2d<T>(in_channels, intermediate_channels, 1, 1, 0, false, option));
        this->add( new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new Conv2d<T>(intermediate_channels, intermediate_channels, 3, stride, 1, false, option));
        this->add( new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        this->add( new Conv2d<T>(intermediate_channels, intermediate_channels * expansion, 1, 1, 0, false, option));
        this->add( new BatchNorm2d<T>());
        if (identity_downsample)
        {
            this->add_identity_layer("Identity_OP_Start");
            this->add(new Conv2d<T>(in_channels, intermediate_channels * 4, 1, stride, 0, false, option));
            this->add(new BatchNorm2d<T>());
            this->add_identity_layer("Identity_OP_Finish");
        }
        this->add_identity_layer("Identity_ADD");
        this->add( new ReLU<T>());
    }
    

    void make_layer(int num_residual_blocks, int intermediate_channels, int stride, string option) {
        bool identity_downsample = false;
        /* vector<SimpleNN<T>> layers; */
        if (stride != 1 || in_channels != intermediate_channels * 4) {
            identity_downsample = true;
        }
        this->add_block(in_channels, intermediate_channels, identity_downsample, stride, option);
        in_channels = intermediate_channels * 4;
        for (int i = 0; i < num_residual_blocks - 1; i++) {
            this->add_block(in_channels, intermediate_channels, false, 1, option);
        }
    }

	void forward(const MatX<T>& X, bool is_training) override {
        MatX<T> identity = X;
        MatX<T> out = X;
        MatX<T> temp = X;
        int i = 0;
		for (int l = 0; l < this->net.size(); l++) {
            if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
                            identity = out; //store identity of current layer

                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
                            out = identity;
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
                        }
                        else if(this->identity_layers_type[i] == "Identity_ADD") {
                            out += identity;
                        }
                    i++;
                    if(i >= this->identity_layers.size()) {
                        break;
                    }
                    }

            }
                this->net[l]->forward(out, is_training);
                out = this->net[l]->output;
            }
        }

	void compile(vector<int> input_shape, Optimizer* optim=nullptr, Loss<T>* loss=nullptr) override
	{
		// set optimizer & loss
		this->optim = optim;
		this->loss = loss;

		// set first & last layer
		this->net.front()->is_first = true;
		this->net.back()->is_last = true;
    
        vector<int> identity = input_shape;
       vector<int> out = input_shape;
        vector<int> temp = input_shape; 


		// set network
        int i = 0;

		for (int l = 0; l < this->net.size(); l++) {
            if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
                            identity = out; //store identity of current layer
                            /* std::cout << "Identity_Store" << std::endl; */
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
                            out = identity;
                            /* std::cout << "Identity_OP_Start" << std::endl; */
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
                            /* std::cout << "Identity_OP_Finish" << std::endl; */
                        }
                    i++;
                    if(i >= this->identity_layers.size()) {
                        break;
                    }
                    }

            }
            /* if (toString(this->net[l]->type) == "LINEAR" || toString(this->net[l]->type) == "BATCHNORM2D" || toString(this->net[l]->type) == "CONV2D") */
            /* { */
            /* std::cout << "Layer: " << l << ", Layer Type: " << toString(this->net[l]->type) << std::endl; */
            /* } */
            this->net[l]->set_layer(out);
            out = this->net[l]->output_shape();
		
        }

		// set Loss layer
		if (loss != nullptr) {
			loss->set_layer(this->net.back()->output_shape());
	}
	}

        
        



};
    

template <typename T>
ResNet<T> ResNet18(int num_classes, string option = "kaiming_uniform", int image_channels = 3) {
    int residual_blocks[4] = {2, 2, 2, 2};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet50(int num_classes, string option = "kaiming_uniform", int image_channels = 3) {
    int residual_blocks[4] = {3, 4, 6, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet101(int num_classes, string option = "kaiming_uniform", int image_channels = 3) {
    int residual_blocks[4] = {3, 4, 23, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}

template <typename T>
ResNet<T> ResNet152(int num_classes, string option = "kaiming_uniform", int image_channels = 3) {
    int residual_blocks[4] = {3, 8, 36, 3};
    return ResNet<T>(residual_blocks, image_channels, num_classes, option);
}







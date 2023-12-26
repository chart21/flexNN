#pragma once
#include "../headers/simple_nn.h"
#include <queue>

using namespace simple_nn;

/*
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1,isCulled=False,isTopThinned=False,isBottomThinned=False):
		super(BasicBlock, self).__init__()

		self.isCulled = isCulled
		self.isTopThinned = isTopThinned
		self.isBottomThinned = isBottomThinned

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isTopThinned):
			self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isBottomThinned):
			self.relu2 = nn.ReLU()

		self.shortcut = nn.Sequential()
			

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		if (not self.isCulled) and (not self.isTopThinned):
			out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += self.shortcut(x)
		if (not self.isCulled) and (not self.isBottomThinned):
			out = self.relu2(out)
		return out



class RESNET(nn.Module):
	def __init__(self, block, num_blocks, num_classes, alpha = 1.0, rho = 1.0, isCulled=[False,False,False,False], isThinned=[False,False]):
		super(ResNet, self).__init__()

		self.alpha = alpha
		self.rho = rho
		self.in_planes = int(64*alpha)
		
		self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,stride=int(1//rho), padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_planes)

		self.layer1 = self._make_layer(block, int(64*alpha),  num_blocks[0], isCulled[0],isThinned,stride=1)
		self.layer2 = self._make_layer(block, int(128*alpha), num_blocks[1], isCulled[1],isThinned,stride=2)
		self.layer3 = self._make_layer(block, int(256*alpha), num_blocks[2], isCulled[2],isThinned,stride=2)
		self.layer4 = self._make_layer(block, int(512*alpha), num_blocks[3], isCulled[3],isThinned,stride=2)
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(int(512*alpha)*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, culled_stages_status,thinning_layer,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride,culled_stages_status,thinning_layer[0],thinning_layer[1]))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		#out = F.relu(self.bn1(self.conv1(x))) 
		out = self.bn1(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out)
		out = torch.flatten(out, 1)
		out = self.fc(out)
		return out

## CIFAR-100 models ###
def DRD_C100_230K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_C100_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_29K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_14K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_12K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_7K(num_classes):
	return ResNet(BasicBlock, [2,1,1,1], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])


## TinyImageNet models ##

def DRD_TINY_918K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_TINY_459K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_393K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,False])

def DRD_TINY_229K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_197K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_98K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,True,False,False], isThinned=[False,True])
*/

    template <typename T>
class ReducedNet : public SimpleNN<T>
{
    private:
        int in_planes;
        int alpha;
        int rho;
        vector<int> identity_layers;
        vector<string> identity_layers_type;
    public:
        ReducedNet(int num_blocks[4], bool isCulled[4], bool isThinned[2], int num_classes, float alpha = 1.0, float rho = 1.0, string option = "kaiming_uniform") {
            const int block_expansion = 1;
            this->alpha = alpha;
            this->rho = rho;
            this->in_planes = int(64*alpha);
            this->add(new Conv2d<T>(3, this->in_planes, 3, int(1/rho), 1, false, option));
            this->add(new BatchNorm2d<T>());
            this->make_layer(num_blocks[0], int(64*alpha), isCulled[0], isThinned[0], isThinned[1], 1, option);
            this->make_layer(num_blocks[1], int(128*alpha), isCulled[1], isThinned[0], isThinned[1], 2, option);
            this->make_layer(num_blocks[2], int(256*alpha), isCulled[2], isThinned[0], isThinned[1], 2, option);
            this->make_layer(num_blocks[3], int(512*alpha), isCulled[3], isThinned[0], isThinned[1], 2, option);
            this->add(new AdaptiveAvgPool2d<T>(1, 1));
            this->add(new Flatten<T>());
            this->add(new Linear<T>(int(512*alpha)*block_expansion, num_classes, option));
        }
		
    void add_identity_layer(string type) {
        this->identity_layers.push_back(this->net.size());
        this->identity_layers_type.push_back(type);
    }

		


        void add_block(int in_planes, int planes, string option, bool isCulled = false, bool isTopThinned = false, bool isBottomThinned = false, int stride = 1) {
            const int expansion = 1;
            this->add_identity_layer("Identity_Store");
            this->add(new Conv2d<T>(in_planes, planes, 3, stride, 1, false, option));
            this->add(new BatchNorm2d<T>());
            if (!isCulled && !isTopThinned) {
                this->add(new ReLU<T>());
            }
            this->add(new Conv2d<T>(planes, planes, 3, 1, 1, false, option));
            this->add(new BatchNorm2d<T>());
            if (stride != 1 || in_planes != planes * expansion) {
                this->add_identity_layer("Identity_OP_Start");
                this->add(new Conv2d<T>(in_planes, planes * expansion, 1, stride, 0, false, option));
                this->add(new BatchNorm2d<T>());
                this->add_identity_layer("Identity_OP_Finish");
            }
            this->add_identity_layer("Identity_ADD");
            if (!isCulled && !isBottomThinned) {
                this->add(new ReLU<T>());
            }
        }

        void make_layer(int num_blocks, int planes, bool isCulled, bool isTopThinned, bool isBottomThinned, int stride, string option) {
            const int block_expansion = 1;
            vector<int> strides;
            strides.push_back(stride);
            for (int i = 0; i < num_blocks - 1; i++) {
                strides.push_back(1);
            }
            for (int stride : strides) {
                this->add_block(this->in_planes, planes, option, isCulled, isTopThinned, isBottomThinned, stride);
                this->in_planes = planes * block_expansion;
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
ReducedNet<T> DRD_C100_230K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_C100_115K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_C100_57K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_C100_49K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,true};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_C100_29K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}

template <typename T>
ReducedNet<T> DRD_C100_14K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_C100_12K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,true};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_C100_7K(int num_classes) {
    int num_blocks[4] = {2,1,1,1};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,true};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_918K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {false,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_459K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_393K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {false,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_229K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_197K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_115K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_98K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 1.0);
}

template <typename T>
ReducedNet<T> DRD_TINY_57K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,false,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 0.5, 0.5);
}

template <typename T>
ReducedNet<T> DRD_TINY_49K(int num_classes) {
    int num_blocks[4] = {2,2,2,2};
    bool isCulled[4] = {true,true,false,false};
    bool isThinned[2] = {true,false};
    return  ReducedNet<T>(num_blocks, isCulled, isThinned, num_classes, 1.0, 0.5);
}





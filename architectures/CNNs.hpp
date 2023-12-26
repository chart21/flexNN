#pragma once
#include "../headers/simple_nn.h"
/*
template<typename T> 
 class VGG : SimpleNN<T>


   VGG(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True)
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): ReLU(inplace=True)
      (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (27): ReLU(inplace=True)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
if classes == 10:
    (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
    (classifier): Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True)
      (1): ReLU(inplace=True)
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=4096, out_features=4096, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=512, out_features=256, bias=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=256, bias=True)
    (4): ReLU()
    (5): Linear(in_features=256, out_features=10, bias=True)
  )
if classes == 100:

)
*/

// Conv2d(int in_channels, int out_channels, int kernel_size, int padding, string option);
// Linear(int in_features, int out_features, string option);
// MaxPool2d(int kernel_size, int stride);
// AvgPool2d(int kernel_size, int stride);

using namespace simple_nn;

template <typename T>
class VGG : public SimpleNN<T>
{
    public:
    VGG(int num_classes)
    {
    this->add(new Conv2d<T>(3,64,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64,64,3,1,1));
    this->add(new ReLU<T>());
    this->add(new AvgPool2d<T>(2,2));
    this->add(new Conv2d<T>(64,128,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128,128,3,1,1));
    this->add(new ReLU<T>());
    this->add(new AvgPool2d<T>(2,2));
    this->add(new Conv2d<T>(128,256,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256,256,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256,256,3,1,1));
    this->add(new ReLU<T>());
    this->add(new AvgPool2d<T>(2,2));
    this->add(new Conv2d<T>(256,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new AvgPool2d<T>(2,2));
    this->add(new Conv2d<T>(512,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512,512,3,1,1));
    this->add(new ReLU<T>());
    this->add(new AvgPool2d<T>(2,2));
    /* net.add(new AdaptiveAvgPool2d<T>(7,7)); */
    
    /* net.add(new Flatten<T>()); */
    /* net.add(new Linear<T>(25088,4096)); */
    /* net.add(new ReLU<T>()); */
    /* net.add(new Linear<T>(4096,4096)); */
    /* net.add(new ReLU<T>()); */
    /* net.add(new Linear<T>(4096,1000)); */

    if(num_classes == 10)
    {
        this->add(new Flatten<T>());
        this->add(new Linear<T>(512,256));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(256,256));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(256,10));
    }
    else if(num_classes == 200)
    {
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(512,512));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(512,200));
    }
    else if(num_classes == 1000)
    {
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(4608,4096));
        this->add(new Linear<T>(4096,4096));
        this->add(new Linear<T>(4096,1000));
    }
    else
    {
        std::cout << "Error: num_classes must be 10, 200, or 1000" << std::endl;
        exit(1);
    }

}

};

template <typename T>       
class LeNet : public SimpleNN<T>
{
    public:
    LeNet(int num_classes)
    {
        this->add(new Conv2d<T>(1,6,5,0,0));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Conv2d<T>(6,16,5,0,0));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(2,2));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(400,120));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(120,84));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(84,num_classes));
    }
};

template <typename T>
class AlexNet : public SimpleNN<T>
{
    public:
    AlexNet(int num_classes)
    {
        this->add(new Conv2d<T>(3,64,11,0,0));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(3,2));
        this->add(new Conv2d<T>(64,192,5,0,0));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(3,2));
        this->add(new Conv2d<T>(192,384,3,0,0));
        this->add(new ReLU<T>());
        this->add(new Conv2d<T>(384,256,3,0,0));
        this->add(new ReLU<T>());
        this->add(new Conv2d<T>(256,256,3,0,0));
        this->add(new ReLU<T>());
        this->add(new AvgPool2d<T>(3,2));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(9216,4096));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(4096,4096));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(4096,num_classes));
    }
};

/*
class AlexNet_32(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 56, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(56, 128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)  # Adjust the size according to your input image size
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout1(self.relu6(self.fc1(x)))
        x = self.dropout2(self.relu7(self.fc2(x)))
        x = self.fc3(x)
        return x
*/

template <typename T>
class AlexNet_32 : public SimpleNN<T>
{
    public:
    AlexNet_32(int num_classes)
    {
        this->add(new Conv2d<T>(3,56,3,1,1));
        this->add(new ReLU<T>());
        this->add(new MaxPool2d<T>(3,2,1));
        this->add(new Conv2d<T>(56,128,5,1,2));
        this->add(new ReLU<T>());
        this->add(new MaxPool2d<T>(3,2,1));
        this->add(new Conv2d<T>(128,192,3,1,1));
        this->add(new ReLU<T>());
        this->add(new Conv2d<T>(192,192,3,1,1));
        this->add(new ReLU<T>());
        this->add(new Conv2d<T>(192,128,3,1,1));
        this->add(new ReLU<T>());
        this->add(new MaxPool2d<T>(3,2,1));
        this->add(new Flatten<T>());
        this->add(new Linear<T>(2048,2048));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(2048,2048));
        this->add(new ReLU<T>());
        this->add(new Linear<T>(2048,num_classes));
    }
};



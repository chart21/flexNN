#pragma once
#include "layer.h"

namespace simple_nn
{
template <typename T>
class Flatten : public Layer<T>
{
  private:
    int batch;
    int channels;
    int height;
    int width;
    int out_block_size;

  public:
    Flatten();
    void set_layer(const vector<int>& input_shape) override;
    void forward(const MatX<T>& prev_out, bool is_training) override;
    void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
    void zero_grad() override;
    vector<int> output_shape() override;
};

template <typename T>
Flatten<T>::Flatten() : Layer<T>(LayerType::FLATTEN)
{
}

template <typename T>
void Flatten<T>::set_layer(const vector<int>& input_shape)
{
    assert(input_shape.size() == 4 && "Flatten::set_layer(const vector<int>&): Must be followed by 2d layer.");
    batch = input_shape[0];
    channels = input_shape[1];
    height = input_shape[2];
    width = input_shape[3];
    out_block_size = batch * channels * height * width;

    this->output.resize(batch, channels * height * width);
    this->delta.resize(batch, channels * height * width);
}

template <typename T>
void Flatten<T>::forward(const MatX<T>& prev_out, bool is_training)
{
    std::copy(prev_out.data(), prev_out.data() + out_block_size, this->output.data());
}

template <typename T>
void Flatten<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
{
    std::copy(this->delta.data(), this->delta.data() + out_block_size, prev_delta.data());
}

template <typename T>
void Flatten<T>::zero_grad()
{
    this->delta.setZero();
}

template <typename T>
vector<int> Flatten<T>::output_shape()
{
    return {batch, channels, height, width};
}
}  // namespace simple_nn

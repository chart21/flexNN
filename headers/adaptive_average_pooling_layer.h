#pragma once
#include "layer.h"

namespace simple_nn
{
    template<typename T>
    class AdaptiveAvgPool2d : public Layer<T>
    {
    private:
        int batch;
        int ch;
        int ih;
        int iw;
        int oh;
        int ow;
        int stride_h;
        int stride_w;
        int kernel_h;
        int kernel_w;

    public:
        AdaptiveAvgPool2d(int target_oh, int target_ow);
        void set_layer(const std::vector<int>& input_shape) override;
        void forward(const MatX<T>& prev_out, bool is_training) override;
        void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
        std::vector<int> output_shape() override;
    };

    template<typename T>
    AdaptiveAvgPool2d<T>::AdaptiveAvgPool2d(int target_oh, int target_ow) :
        Layer<T>(LayerType::ADAPTIVEAVGPOOL2D), oh(target_oh), ow(target_ow) {}

    template<typename T>
void AdaptiveAvgPool2d<T>::set_layer(const std::vector<int>& input_shape)
{
    batch = input_shape[0];
    ch = input_shape[1];
    ih = input_shape[2];
    iw = input_shape[3];

    // Calculate stride and kernel size for both height and width
    stride_h = std::ceil(static_cast<float>(ih) / static_cast<float>(oh));
    stride_w = std::ceil(static_cast<float>(iw) / static_cast<float>(ow));

    kernel_h = ih - (oh - 1) * stride_h;
    kernel_w = iw - (ow - 1) * stride_w;

    // Resize output and delta matrices based on new output dimensions
    this->output.resize(batch * ch, oh * ow);
    this->delta.resize(batch * ch, oh * ow);
}

template<typename T>
void AdaptiveAvgPool2d<T>::forward(const MatX<T>& prev_out, bool is_training)
{
#if TRUNC_DELAYED == 1
        if(delayed)
            #if TRUNC_APPROACH == 0
                trunc_pr_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
            #elif TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4
                trunc_2k_in_place(const_cast<T*>(prev_out.data()), prev_out.size(),false);
            #elif TRUNC_APPROACH == 2
                trunc_exact_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
            #elif TRUNC_APPROACH == 3
                trunc_exact_opt_in_place(const_cast<T*>(prev_out.data()), prev_out.size());
            #endif
        delayed = false;
#endif
    T* out = this->output.data();
    const T* pout = prev_out.data();
    auto denominators = new int[this->output.size()];
    bool all_one = true;

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < ch; ++c) {
            for (int h = 0; h < oh; ++h) {
                for (int w = 0; w < ow; ++w) {
                    // Calculate the start and end indices for the kernel
                    int h_start = h * stride_h;
                    int h_end = std::min(h_start + kernel_h, ih);
                    int w_start = w * stride_w;
                    int w_end = std::min(w_start + kernel_w, iw);

                    // Initialize the sum for the current region
                    T sum = 0;
                    for (int y = h_start; y < h_end; ++y) {
                        for (int x = w_start; x < w_end; ++x) {
                            int idx = x + iw * (y + ih * (c + ch * n));
                            sum += pout[idx];
                        }
                    }
                    // Compute the average and assign it to the output
                    int out_idx = w + ow * (h + oh * (c + ch * n));
                    out[out_idx] = sum;
                    int denominator = (h_end - h_start) * (w_end - w_start);
                    if(denominator != 1)
                        all_one = false;
                    denominators[out_idx] = denominator;
                }
            }
        }
    }
if(all_one)
{
print_online("All denominators in AdaptiveAvgPool2d are 1, skipping division");
delete[] denominators;
return;
}

for (int i = 0; i < this->output.size(); i++)
{
#if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
        if(denominators[i] <= 1)
            continue;
        if (((denominators[i]) & (denominators[i] - 1)) == 0) // if power of 2
            out[i] = out[i].prepare_div_exp2(denominators[i]);
        else
            out[i] *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/(denominators[i]));
/* #elif TRUNC_THEN_MULT == 0 */
#else
            out[i] = out[i].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/(denominators[i])));
#endif
}
#if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
T::communicate();
for (int i = 0; i < this->output.size(); i++)
{
        if(denominators[i] > 1)
            out[i].complete_public_mult_fixed();
}
#endif

delete[] denominators;

#if TRUNC_APPROACH == 1 
     trunc_2k_in_place(out, this->output.size(),true); // trunc before mult with demoniator */
 #elif TRUNC_APPROACH == 2 
     trunc_exact_in_place(out, this->output.size()); 
#elif TRUNC_APPROACH == 3
     trunc_exact_opt_in_place(out, this->output.size(),true);
 #endif 


}

    template<typename T>
    void AdaptiveAvgPool2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
    {
        // Implement backward pass based on adaptive pooling logic
        // ...
    }

    template<typename T>
    std::vector<int> AdaptiveAvgPool2d<T>::output_shape()
    {
        return { batch, ch, oh, ow };
    }
}



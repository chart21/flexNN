#pragma once
#include "common.h"

namespace simple_nn
{
template <typename T>
class Loss
{
  public:
    int batch;
    int n_label;

  public:
    Loss() : batch(0), n_label(0) {}

    void set_layer(const vector<int>& input_shape)
    {
        assert(input_shape.size() == 2 && "Loss::set(const vector<int>): Output layer must be linear.");
        batch = input_shape[0];
        n_label = input_shape[1];
    }

    virtual T calc_loss(const MatX<T>& prev_out, const VecXi& labels, MatX<T>& prev_delta) = 0;
};

template <typename T>
class MSELoss : public Loss<T>
{
  public:
    MSELoss() : Loss<T>() {}

    T calc_loss(const MatX<T>& prev_out, const VecXi& labels, MatX<T>& prev_delta) override
    {
        T loss_batch(0);
        T loss = (0);
        prev_delta = prev_out;
        for (int n = 0; n < this->batch; n++)
        {
            prev_delta(n, labels[n]) -= 1;
            for (int i = 0; i < this->n_label; i++)
            {
                loss = prev_delta(n, i);
                /* loss_batch += 0.5f * loss * loss; */
                loss_batch += (loss * loss) / 2;
            }
            // loss_batch += 0.5f * prev_delta.row(n).pow(2).sum();
        }
        return loss_batch / this->batch;
    }
};

template <typename T>
class CrossEntropyLoss : public Loss<T>
{
  public:
    CrossEntropyLoss() : Loss<T>() {}

    T calc_loss(const MatX<T>& prev_out, const VecXi& labels, MatX<T>& prev_delta)
    {
        T loss_batch(0.f);
        prev_delta = prev_out;
        for (int n = 0; n < this->batch; n++)
        {
            int answer_idx = labels[n];
            prev_delta(n, answer_idx) -= 1.f;
            loss_batch -= prev_out(n, answer_idx).log();
        }
        return loss_batch / this->batch;
    }
};
}  // namespace simple_nn

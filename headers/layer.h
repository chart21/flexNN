#pragma once
#include "common.h"

namespace simple_nn
{
	enum class LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
        ADAPTIVEAVGPOOL2D,
		ACTIVATION,
		BATCHNORM1D,
		BATCHNORM2D,
		FLATTEN
	};
    
    std::string toString(LayerType type) {
    switch (type) {
    case LayerType::LINEAR: return "LINEAR";
    case LayerType::CONV2D: return "CONV2D";
    case LayerType::MAXPOOL2D: return "MAXPOOL2D";
    case LayerType::AVGPOOL2D: return "AVGPOOL2D";
    case LayerType::ADAPTIVEAVGPOOL2D: return "ADAPTIVEAVGPOOL2D";
    case LayerType::ACTIVATION: return "ACTIVATION";
    case LayerType::BATCHNORM1D: return "BATCHNORM1D";
    case LayerType::BATCHNORM2D: return "BATCHNORM2D";
    case LayerType::FLATTEN: return "FLATTEN";
    default: return "Unknown LayerType";
    }
}

    template<typename T>
	class Layer
	{
	public:
		LayerType type;
		bool is_first;
		bool is_last;
		MatX<T> output;
		MatX<T> delta;
	public:
		Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
		virtual void set_layer(const vector<int>& input_shape) = 0;
		virtual void forward(const MatX<T>& prev_out, bool is_training = true) = 0;
		virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) = 0;
		virtual void update_weight(float lr, float decay) { return; }
		virtual void zero_grad() { return; }
		virtual vector<int> output_shape() = 0;
	};
}

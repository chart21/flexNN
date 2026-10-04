#pragma once
#include "../headers/simple_nn.h"
#include <array>

using namespace simple_nn;

// === PYTORCH Equivalent, CPP Version below ===

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
protected:
    int in_channels;
    vector<int> identity_layers;
    vector<string> identity_layers_type;
public:
    ResNet() {}
    ResNet(int residual_blocks[4], int image_channels, int num_classes, string option = "kaiming_uniform") {
        in_channels = 64;
        this->add(new Conv2d<T>(image_channels, 64, 7, 2, 3, false, option));
        this->add(new BatchNorm2d<T>());
        this->add( new ReLU<T>());
        /* this->add( new MaxPool2d<T>(3, 2, 1)); */
        this->add( new AvgPool2d<T>(3, 2, 1)); // Replaced MaxPool with AvgPool
        this->make_layer( residual_blocks[0], 64, 1, option);
        this->make_layer( residual_blocks[1], 128, 2, option);
        this->make_layer( residual_blocks[2], 256, 2, option);
        this->make_layer( residual_blocks[3], 512, 2, option);
        this->add( new AdaptiveAvgPool2d<T>(1, 1));
        this->add( new Flatten<T>());
        this->add(new Linear<T>(512 * 4, num_classes, option));
    }

    void add_identity_layer(string type) {
        this->identity_layers.push_back(this->net.size());
        this->identity_layers_type.push_back(type);
    }

    vector<int> residual_sums() const {
        vector<int> at;
        for (size_t k = 0; k < this->identity_layers.size(); k++)
            if (this->identity_layers_type[k] == "Identity_ADD")
                at.push_back(this->identity_layers[k]);
        return at;
    }

    // A2B bake, residual sums (after mark_baked_relu_inputs): number the sums and mark each one's other addend's
    // producer (the partner is the layer computed last), whose masks P1 then commits as well (see g_residual_sums).
    // Replays the Identity_* events of forward() on layer indices.
    void mark_residual_producers() {
#if A2B_CONV_BAKE_ACTIVE && A2B_BAKE_RESIDUAL == 1
        int out_src = -1, identity_src = -1, temp_src = -1, k = 0;  // the layer whose output each holds (-1: input)
        size_t i = 0;
        for (int l = 0; l < (int)this->net.size(); l++) {
            for (; i < this->identity_layers.size() && this->identity_layers[i] == l; i++) {
                const string& type = this->identity_layers_type[i];
                if (type == "Identity_Store")
                    identity_src = out_src;
                else if (type == "Identity_OP_Start") {
                    temp_src = out_src;
                    out_src = identity_src;
                }
                else if (type == "Identity_OP_Finish") {
                    identity_src = out_src;
                    out_src = temp_src;
                }
                else if (type == "Identity_ADD") {
                    auto* relu = dynamic_cast<ReLU<T>*>(this->net[l]);
                    const int other = identity_src == l - 1 ? out_src : identity_src;
                    if (relu && relu->input_residual && other >= 0 && other != l - 1) {
                        relu->residual_k = k;
                        int producer = 0;
                        if (auto* conv = dynamic_cast<Conv2d<T>*>(this->net[other]); conv && !conv->bake_output) {
                            conv->residual_producer_k = k;
                            producer = 1;
                        }
                        else if (auto* fc = dynamic_cast<Linear<T>*>(this->net[other]); fc && !fc->bake_output) {
                            fc->residual_producer_k = k;
                            producer = 1;
                        }
                        else if (auto* prelu = dynamic_cast<ReLU<T>*>(this->net[other]); prelu && !prelu->fused_into_maxpool()) {
                            prelu->identity_k = k;
                            producer = 2;
                        }
                        residual_sum(k).producer = producer;
                    }
                    k++;
                }
            }
            out_src = l;
        }
#endif
    }
 
    // The layers whose outputs the residual sums read (replaying the Identity_* events of forward() on layer indices,
    // as mark_residual_producers): their ReLUs stay revealed to both parties (mark_one_way_relus)
    vector<int> residual_operands() const {
        vector<int> ops;
        int out_src = -1, identity_src = -1, temp_src = -1;
        size_t i = 0;
        for (int l = 0; l <= (int)this->net.size(); l++) {
            for (; i < this->identity_layers.size() && this->identity_layers[i] == l; i++) {
                const string& type = this->identity_layers_type[i];
                if (type == "Identity_Store")
                    identity_src = out_src;
                else if (type == "Identity_OP_Start") {
                    temp_src = out_src;
                    out_src = identity_src;
                }
                else if (type == "Identity_OP_Finish") {
                    identity_src = out_src;
                    out_src = temp_src;
                }
                else if (type == "Identity_ADD" || type == "Identity_CAT" || type == "Identity_MUL") {
                    ops.push_back(out_src);
                    ops.push_back(identity_src);
                }
                else
                    ops.push_back(out_src), ops.push_back(identity_src);  // unknown event: keep both revealed
            }
            out_src = l;
        }
        return ops;
    }

    // RES_MERGE_ACTIVE: the residual sums whose partner (the addend computed last) is a conv and whose other addend P1
    // never reads except through the sum: a conv whose output only the sum reads (merge_skip: not sent), or a ReLU whose
    // other readers are convs / FC layers (one-way). The partner's message carries both addends (g_res_merge_add, set by
    // forward()), and P1 zeroes the other addend before the sum (res_merge). Returns the merged sums' ReLU addends.
    vector<char> res_merge;  // per Identity_* event: an Identity_ADD whose sum is merged
    vector<int> mark_residual_merges() {
        vector<int> relus;
        res_merge.assign(this->identity_layers.size(), 0);
#if RES_MERGE_ACTIVE
        const int L = (int)this->net.size();
        vector<int> in_src(L, -1);  // the layer whose output each layer reads (-1: the network input)
        vector<std::array<int, 3>> sums;  // (event, partner, other)
        int out_src = -1, identity_src = -1, temp_src = -1;
        size_t i = 0;
        for (int l = 0; l < L; l++) {
            for (; i < this->identity_layers.size() && this->identity_layers[i] == l; i++) {
                const string& type = this->identity_layers_type[i];
                if (type == "Identity_Store")
                    identity_src = out_src;
                else if (type == "Identity_OP_Start") {
                    temp_src = out_src;
                    out_src = identity_src;
                }
                else if (type == "Identity_OP_Finish") {
                    identity_src = out_src;
                    out_src = temp_src;
                }
                else if (type == "Identity_ADD") {
                    if (out_src == l - 1 && identity_src != l - 1)
                        sums.push_back({(int)i, l - 1, identity_src});
                    else if (identity_src == l - 1 && out_src != l - 1)
                        sums.push_back({(int)i, l - 1, out_src});
                    out_src = -2 - (int)i;  // the sum: the next layer reads neither addend
                }
                else
                    return relus;  // an unknown event: no merging
            }
            in_src[l] = out_src;
            out_src = l;
        }
        for (auto [event, partner, other] : sums) {
            if (other < 0 || dynamic_cast<Conv2d<T>*>(this->net[partner]) == nullptr)
                continue;
            int readers = 0, linear_readers = 0, sums_reading = 0;
            for (int x = 0; x < L; x++)
                if (in_src[x] == other) {
                    readers++;
                    int n = x;  // the layer behind any average poolings
                    while (n < L && (this->net[n]->type == LayerType::AVGPOOL2D ||
                                     this->net[n]->type == LayerType::ADAPTIVEAVGPOOL2D))
                        n = n + 1 < L && in_src[n + 1] == n ? n + 1 : L;
                    linear_readers += n < L && (this->net[n]->type == LayerType::CONV2D ||
                                                this->net[n]->type == LayerType::LINEAR);
                }
            for (auto& s : sums)
                sums_reading += s[2] == other;
            if (sums_reading != 1)
                continue;
            if (auto* conv = dynamic_cast<Conv2d<T>*>(this->net[other]); conv && readers == 0)
                conv->merge_skip = true;
            else if (auto* relu = dynamic_cast<ReLU<T>*>(this->net[other]);
                     relu && !relu->fused_into_maxpool() && readers == linear_readers)
                relus.push_back(other);
            else
                continue;
            dynamic_cast<Conv2d<T>*>(this->net[partner])->merge_partner = true;
            res_merge[event] = 1;
        }
#endif
        return relus;
    }

    // residual_operands() without the merged sums' ReLU addends
    vector<int> unmerged_residual_operands(const vector<int>& merged) const {
        vector<int> ops;
        for (int o : residual_operands())
            if (std::find(merged.begin(), merged.end(), o) == merged.end())
                ops.push_back(o);
        return ops;
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
#if TRUNC_DELAYED == 1
        // The block input as it is, for a downsample branch: a linear layer on the unscaled input gives the scale of
        // the main branch's pending output directly (instead of truncating identity * 2^FRACTIONAL first)
        MatX<T> identity_in = X;
        bool identity_in_delayed = false;
        bool temp_delayed = false;  // the main branch's pending truncation while the downsample branch runs
#endif
        int i = 0;
		for (int l = 0; l < this->net.size(); l++) {
            if(this->identity_layers.size() != 0 && i < this->identity_layers.size()) {
                while(this->identity_layers[i] == l)  { 
                        if(this->identity_layers_type[i] == "Identity_Store") {
#if TRUNC_DELAYED == 0
                            identity = out; //store identity of current layer
#else
if(delayed)
    identity = out;
else
{
    /* identity = out.mult_public(UINT_TYPE(1) << FRACTIONAL); */
    identity = out;
    for(int i = 0; i < identity.size(); i++) {
        identity.data()[i] = identity.data()[i].mult_public(UINT_TYPE(1) << FRACTIONAL);
    }
}
identity_in = out;
identity_in_delayed = delayed;
#endif
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Start") {
                            //network starts operating on identity, storing last output
                            temp = out; 
#if TRUNC_DELAYED == 1
                            temp_delayed = delayed;
                            out = identity_in;  // the downsample conv's output has the main branch's scale
                            delayed = identity_in_delayed;
#else
                            out = identity;
#endif
                        }
                        else if(this->identity_layers_type[i] == "Identity_OP_Finish") {
                            //network finished processing identity, loading back last output
                            identity = out;
                            out = temp;
#if TRUNC_DELAYED == 1
                            // back to the main branch's state: with the downsample branch at the block's start
                            // (Cheetah_ResNet) the main branch has not computed anything yet, and its first conv must
                            // not truncate the block input
                            delayed = temp_delayed;
#endif
                        }
                        else if(this->identity_layers_type[i] == "Identity_ADD") {
#if RES_MERGE_ACTIVE && PARTY == 1
                            // a merged sum: the partner's message carried the other addend (mark_residual_merges);
                            // with a downsample branch finishing here the partner is the identity
                            if (res_merge[i]) {
                                bool finished = false;
                                for (size_t k = 0; k < i; k++)
                                    finished |= this->identity_layers[k] == l && this->identity_layers_type[k] == "Identity_OP_Finish";
                                MatX<T>& other = finished ? out : identity;
                                for (Eigen::Index e = 0; e < other.size(); e++)
                                    other.data()[e].zero_m();
                            }
#endif
                            out += identity;
                        }
                    i++;
                    if(i >= this->identity_layers.size()) {
                        break;
                    }
                    }

            }
                start_layer_stats(toString(this->net[l]->type), l);
                /* start_timer(); */
#if A2B_CONV_BAKE_ACTIVE && A2B_BAKE_RESIDUAL == 1
                // A residual sum at l + 1 whose partner is this conv/FC (the addend computed last): publish the other
                // addend's masks, so that this layer's baked masks are lz - those (g_bake_res_l). With a downsample
                // branch finishing at l + 1, this layer's output becomes the identity and the other addend is temp.
                std::vector<DATATYPE> res_l;
                if (current_phase != PHASE_INIT &&
                    (this->net[l]->type == LayerType::CONV2D || this->net[l]->type == LayerType::LINEAR)) {
                    const MatX<T>* other = nullptr;
                    bool finish = false;
                    for (size_t k = i; k < this->identity_layers.size() && this->identity_layers[k] == l + 1; k++) {
                        if (this->identity_layers_type[k] == "Identity_OP_Finish")
                            finish = true;
                        else if (this->identity_layers_type[k] == "Identity_ADD") {
                            other = finish ? &temp : &identity;
                            break;
                        }
                    }
                    if (other) {
                        res_l.resize(other->size());
                        for (Eigen::Index e = 0; e < other->size(); e++)
                            res_l[e] = other->data()[e].get_share().get_mask();
                        g_bake_res_l = res_l.data();
                        if (l + 1 < (int)this->net.size())
                            if (auto* relu = dynamic_cast<ReLU<T>*>(this->net[l + 1]))
                                g_bake_res_k = relu->residual_k;
                    }
                }
#endif
#if RES_MERGE_ACTIVE
                // residual merge (mark_residual_merges): a conv computed first sends nothing; P0 adds the other
                // addend's masked values to the partner's message (the same buffer as the baked masks above)
                std::vector<DATATYPE> merge_add;
                if (auto* conv = dynamic_cast<Conv2d<T>*>(this->net[l])) {
                    g_res_merge_skip = conv->merge_skip;
#if PARTY == 0
                    if (conv->merge_partner && current_phase == PHASE_LIVE) {
                        const MatX<T>* other = nullptr;
                        bool finish = false;
                        for (size_t k = i; k < this->identity_layers.size() && this->identity_layers[k] == l + 1; k++) {
                            if (this->identity_layers_type[k] == "Identity_OP_Finish")
                                finish = true;
                            else if (this->identity_layers_type[k] == "Identity_ADD") {
                                other = finish ? &temp : &identity;
                                break;
                            }
                        }
                        merge_add.resize(other->size());
                        for (Eigen::Index e = 0; e < other->size(); e++)
                            merge_add[e] = other->data()[e].get_m();
                        g_res_merge_add = merge_add.data();
                    }
#endif
                }
#endif
                this->net[l]->forward(out, is_training);
#if RES_MERGE_ACTIVE
                g_res_merge_skip = false;
                g_res_merge_add = nullptr;
#endif
#if A2B_CONV_BAKE_ACTIVE && A2B_BAKE_RESIDUAL == 1
                g_bake_res_l = nullptr;
                g_bake_res_k = -1;
#endif
                out = this->net[l]->output;
                /* stop_timer(toString(this->net[l]->type)); */
                stop_layer_stats(l);
            #if IS_TRAINING == 0
            if (l > 0 && !g_mask_pass)  // the mask-only forward (A2B_BAKE_MASK_PASS) is followed by the real one
                delete this->net[l - 1];
            #endif
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
            this->net[l]->set_layer(out);
            out = this->net[l]->output_shape();
		
        }
        this->fuse_relu_pools();
        this->mark_baked_relu_inputs(residual_sums());
        mark_residual_producers();
        this->mark_one_way_relus(unmerged_residual_operands(mark_residual_merges()));
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



template <typename T>
class Cheetah_ResNet : public ResNet<T>
{
    private:
        vector<int> increase_size;
        vector<int> increased_sized;
    public:
    Cheetah_ResNet(int num_classes, string option = "kaiming_uniform", int image_channels = 3) 
    {
    // Initial convolution block
    this->add(new Conv2d<T>(3, 64, 7, 2, 0)); // First conv: 3->64, 7x7, stride 2, VALID padding
    this->add(new AvgPool2d<T>(3, 2, 1));     // Maxpool: 3x3, stride 2, VALID padding
    this->add(new BatchNorm2d<T>());        // BatchNorm on 64 channels
    this->add(new ReLU<T>());
    this->add_identity_layer("Identity_Store");
    this->add_identity_layer("Identity_OP_Start");

    // First set of residual blocks (64->256)
    this->add(new Conv2d<T>(64, 256, 1, 1, 0));  // Conv #2: 64->256, 1x1
    this->add_identity_layer("Identity_OP_Finish");
     
    this->add(new Conv2d<T>(64, 64, 1, 1, 0));   // Conv #3: 64->64, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 64, 3, 1, 1));   // Conv #4: 64->64, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 256, 1, 1, 0));  // Conv #5: 64->256, 1x1
    this->add_identity_layer("Identity_ADD");
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Second residual block
    this->add(new Conv2d<T>(256, 64, 1, 1, 0));  // Conv #6: 256->64, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 64, 3, 1, 1));   // Conv #7: 64->64, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 256, 1, 1, 0));  // Conv #8: 64->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Third residual block
    this->add(new Conv2d<T>(256, 64, 1, 1, 0));  // Conv #9: 256->64, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 64, 3, 1, 1));   // Conv #10: 64->64, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(64, 256, 1, 1, 0));  // Conv #11: 64->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add_identity_layer("Identity_Store");
    this->add_identity_layer("Identity_OP_Start");

    // Transition to next block with stride 2
    this->add(new Conv2d<T>(256, 512, 1, 2, 0));  // Conv #12: 256->512, 1x1, stride 2
    this->add_identity_layer("Identity_OP_Finish"); 
    this->add(new Conv2d<T>(256, 128, 1, 1, 0));  // Conv #13: 256->128, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    plan_increase_size(NUM_INPUTS, 128, 58, 58); 

    this->add(new Conv2d<T>(128, 128, 3, 2, 0));  // Conv #14: 128->128, 3x3, stride 2, VALID padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 512, 1, 1, 0));  // Conv #15: 128->512, 1x1
    // this->add_identity_layer("Identity_ADD"); 
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Next set of residual blocks (512->128->512)
    this->add(new Conv2d<T>(512, 128, 1, 1, 0));  // Conv #16: 512->128, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 128, 3, 1, 1));  // Conv #17: 128->128, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 512, 1, 1, 0));  // Conv #18: 128->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(512, 128, 1, 1, 0));  // Conv #19: 512->128, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 128, 3, 1, 1));  // Conv #20: 128->128, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 512, 1, 1, 0));  // Conv #21: 128->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(512, 128, 1, 1, 0));  // Conv #22: 512->128, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 128, 3, 1, 1));  // Conv #23: 128->128, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(128, 512, 1, 1, 0));  // Conv #24: 128->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add_identity_layer("Identity_Store");
    this->add_identity_layer("Identity_OP_Start");

    // Transition to next block with stride 2
    this->add(new Conv2d<T>(512, 1024, 1, 2, 0)); // Conv #25: 512->1024, 1x1, stride 2
    this->add_identity_layer("Identity_OP_Finish"); 
    this->add(new Conv2d<T>(512, 256, 1, 1, 0));  // Conv #26: 512->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    plan_increase_size(NUM_INPUTS, 256, 30, 30); 
    this->add(new Conv2d<T>(256, 256, 3, 2, 0));  // Conv #27: 256->256, 3x3, stride 2, VALID padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #28: 256->1024, 1x1
    // this->add_identity_layer("Identity_ADD"); 
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Next set of residual blocks (1024->256->1024)
    this->add(new Conv2d<T>(1024, 256, 1, 1, 0)); // Conv #29: 1024->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 256, 3, 1, 1));  // Conv #30: 256->256, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #31: 256->1024, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(1024, 256, 1, 1, 0)); // Conv #32: 1024->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 256, 3, 1, 1));  // Conv #33: 256->256, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #34: 256->1024, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(1024, 256, 1, 1, 0)); // Conv #35: 1024->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 256, 3, 1, 1));  // Conv #36: 256->256, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #37: 256->1024, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(1024, 256, 1, 1, 0)); // Conv #38: 1024->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 256, 3, 1, 1));  // Conv #39: 256->256, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #40: 256->1024, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(1024, 256, 1, 1, 0)); // Conv #41: 1024->256, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 256, 3, 1, 1));  // Conv #42: 256->256, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(256, 1024, 1, 1, 0)); // Conv #43: 256->1024, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add_identity_layer("Identity_Store");
    this->add_identity_layer("Identity_OP_Start");

    // Transition to final block with stride 2
    this->add(new Conv2d<T>(1024, 2048, 1, 2, 0)); // Conv #44: 1024->2048, 1x1, stride 2
    this->add_identity_layer("Identity_OP_Finish"); 
    this->add(new Conv2d<T>(1024, 512, 1, 1, 0));  // Conv #45: 1024->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    plan_increase_size(NUM_INPUTS, 512, 16, 16);
    this->add(new Conv2d<T>(512, 512, 3, 2, 0));   // Conv #46: 512->512, 3x3, stride 2, VALID padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512, 2048, 1, 1, 0));  // Conv #47: 512->2048, 1x1
    // this->add_identity_layer("Identity_ADD"); 
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Final residual blocks (2048->512->2048)
    this->add(new Conv2d<T>(2048, 512, 1, 1, 0));  // Conv #48: 2048->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512, 512, 3, 1, 1));   // Conv #49: 512->512, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512, 2048, 1, 1, 0));  // Conv #50: 512->2048, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // Another residual block
    this->add(new Conv2d<T>(2048, 512, 1, 1, 0));  // Conv #51: 2048->512, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512, 512, 3, 1, 1));   // Conv #52: 512->512, 3x3, SAME padding
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());
    this->add(new Conv2d<T>(512, 2048, 1, 1, 0));  // Conv #53: 512->2048, 1x1
    this->add(new BatchNorm2d<T>());
    this->add(new ReLU<T>());

    // // Final pooling and classification
    // this->add(new AvgPool2d<T>(7, 7, 0));          // Global average pooling: 7x7
        this->add( new AdaptiveAvgPool2d<T>(1, 1));
        this->add( new Flatten<T>());
        this->add(new Linear<T>(512 * 4, num_classes));
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
        this->net[l]->set_layer(out);
        if(increase_size[0] == l) {
            auto& layer = this->net[l]->output;
            incr_size(layer, increased_sized[0], increased_sized[1], increased_sized[2], increased_sized[3]);
            if(increased_sized[1] > 0)
                out = {increased_sized[0], increased_sized[1], increased_sized[2], increased_sized[3]};
            else
                out = {increased_sized[0], increased_sized[2]};
            increase_size.erase(increase_size.begin());
            increased_sized.erase(increased_sized.begin(), increased_sized.begin() + 4);
        }
        else
        {
            out = this->net[l]->output_shape();
        }
                
    
    }
    this->fuse_relu_pools();
    this->mark_baked_relu_inputs(this->residual_sums());
    this->mark_residual_producers();
    this->mark_one_way_relus(this->unmerged_residual_operands(this->mark_residual_merges()));
    // set Loss layer
    if (loss != nullptr) {
        loss->set_layer(this->net.back()->output_shape());
}
}
void plan_increase_size(int n, int ic, int ih, int iw)
{
    increase_size.push_back(this->net.size() - 1);
    increased_sized.push_back(n);
    increased_sized.push_back(ic);
    increased_sized.push_back(ih);
    increased_sized.push_back(iw);
}

template <typename L>
void incr_size(L &layer, int n, int ic, int ih, int iw)
{
    layer.resize(n * ic, ih * iw);
}

};




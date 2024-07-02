# FlexNN

FlexNN is a templated C++ Neural Network engine forked from [SimpleNN](https://github.com/stnamjef/SimpleNN), [MIT LICENSE](https://raw.githubusercontent.com/stnamjef/SimpleNN/master/LICENSE).
- FlexNN implements all layers required to evaluate ResNets and Convolutional Neural Networks.
- Models and datasets can be imported from PyTorch from .bin files using [PyGEON](https://github.com/chart21/pygeon). Save the models and datasets exported by PyGEON to `model_zoo/` and `dataset/` respectively.
- New Model architectures can be added to `architectures/`.
- Models and datasets can be evaluated with various Secure Multiparty Computation protocols using [HPMPC](https://github.com/chart21/hpmpc).

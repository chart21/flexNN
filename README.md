# FlexNN

FlexNN is a templated C++ inference engine forked from [SimpleNN](https://github.com/stnamjef/SimpleNN).
- Implements all layers required to evaluate ResNets and Convolutional Neural Networks.
- Models and datasets can be imported from PyTorch from as .bin files using [PyGEON](https://github.com/chart21/pygeon). Save the files exported by PyGEON to `model_zoo/` or `dataset/`.
- New Model architectures can be added in `architectures/`

# TMNet
This repository contains the source codes for the paper 
[Deep Mesh Reconstruction from Single RGB Images via Topology Modification Networks](https://arxiv.org/abs/1909.00321).
As we reorganized our code based on new python and pytorch versions, some hyper-parameters and results are slightly different from the paper.

### Citing this work

If you find this work useful in your research, please consider citing:

```
@inproceedings{pan2019deep,
  title={Deep Mesh Reconstruction from Single RGB Images via Topology Modification Networks},
  author={Pan, Junyi and Han, Xiaoguang and Chen, Weikai and Tang, Jiapeng and Jia, Kui},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9964--9973},
  year={2019}
}
```
# Install

### Clone the repo and install dependencies

This implementation uses [Pytorch 1.0.0](http://pytorch.org/). 

```shell
## Download the repository
git clone https://github.com/jnypan/TMNet.git
cd TMNet
## Create python env with relevant packages
conda create --name tmnet python=3.7
source activate tmnet
pip install pandas visdom
cd ./extension
python setup.py install
```

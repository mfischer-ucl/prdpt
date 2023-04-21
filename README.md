## Plateau-reduced Differentiable Path Tracing

This is the offical code repository for our paper

**Plateau-reduced Differentiable Path Tracing [CVPR2 2023]**

by [Michael Fischer](https://mfischer-ucl.github.io) and [Tobias Ritschel](https://www.homepages.ucl.ac.uk/~ucactri). 
For more information, make sure to check out the [paper](https://arxiv.org/pdf/2211.17263.pdf) 
and [project page](https://mfischer-ucl.github.io/prdpt/).

<p align="center">
 <img src="./assets/optim.gif" width="75%" height="37.5%" />
</p>

___
### Installation

Clone the repository, then create a new conda environment, activate it and install the dependencies.  
```
git clone https://github.com/mfischer-ucl/prdpt 
cd prdpt 
conda create -n prdpt python=3.9 
conda activate prdpt
pip install -r requirements.txt
```

Tested with Python 3.9, PyTorch 1.11 and CUDA 11.7 on Ubuntu 20.04.4 x64 and an NVIDIA RTX3000 series GPU.   
___
### Usage 

This repository uses [Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/#) to render the images. The core functionality, 
i.e., the importance sampler and kernel functions, reside in `utils_fns.py`.  We prodive the sphere example on top under 
`examples/sphere_example.py`. The main function is `run_optimization.py  

#### Running your own experiments 
This is easy to do. Simply provide a Mitsuba scenefile (.xml), 
load it and get the information needed (cf. `setup_shadowscene.py`), 
and then write your own `update_fn` to update the scene parameters. Good to go! 

___
### License 
This code is licensed under the MIT license. 
___
### Citation

If you find our work useful or plan to (re-) use parts of it in your own projects, please include the following citation:

```
@article{fischer2022plateau,
  title={Plateau-reduced Differentiable Path Tracing},
  author={Fischer, Michael and Ritschel, Tobias},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
} 
```

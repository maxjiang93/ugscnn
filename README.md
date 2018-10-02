## UGSCNN: Spherical CNNs on Unstructured Grids
 
By: [Chiyu "Max" Jiang](http://cfd.me.berkeley.edu/people/chiyu-max-jiang/), [Jingwei Huang](http://stanford.edu/~jingweih/), [Karthik Kashinath](http://www.nersc.gov/about/nersc-staff/data-analytics-services/karthik-kashinath/), [Prabhat](http://www.nersc.gov/about/nersc-staff/data-analytics-services/prabhat/), [Philip Marcus](http://www.me.berkeley.edu/people/faculty/philip-s-marcus), [Matthias Niessner](http://niessnerlab.org/)
![teaser](doc/ugscnn_teaser.png "UGSCNN_teaser" | width=200)

### Introduction
This repository is based on our paper: UGSCNN: Spherical CNNs on Unstructured Grids. #TODO:LINK. The Project Webpage #TODO:LINK presents an overview of the project. 

In this project, we present an alternative convolution kernel for deploying CNNs on unstructured grids, using parameterized differential operators. More specifically we evaluate this method for the spherical domain that is discritized using the icosahedral spherical mesh. Our unique convolution kernel parameterization scheme achieves high parameter efficiency compared to competing methods. We evaluate our model for classification as well as semantic segmentation tasks. Please see `experiments/` for detailed examples.

Our deep learning code base is written using [PyTorch](https://pytorch.org/) in Python 3, in conjunction with standard ML packages such as [Scikit-Learn](http://scikit-learn.org/stable/) and [Numpy](http://www.numpy.org/).

### Instructions
To acquire the mesh files used in this project, run the provided script `gen_mesh.py`. 
```bash
python gen_mesh
```
To locally generate the mesh files, the [Libigl](http://libigl.github.io/libigl/) library is required. Libigl is mainly used for computing the Laplacian and Derivative matrices that are stored in the pickle files. Alternatively the script will download precomputed pickles if the library is not available.

To run experiments, please find details instructions in under individual experiments in [`experiments`](experiments). For most experiments, simply running the script `run.sh` is sufficient to start the training process.

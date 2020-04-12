# PoET-BiN

This repo contains the implementation for [PoET-BiN: Power-Efficient Tiny Binary Neurons](https://proceedings.mlsys.org/papers/2020/78) for the MNIST, CIFAR-10 and SVHN datasets.

There is a separate script to run each dataset present inside the respective folder.

Software requirements:
* Python 3
* CUDA 9
* PyTorch 1.0

Hardware requirements:
* Nvidia GPU to train the model
* Multi-core CPU (I used 32, more the better) with atleast 32 GB RAM (I used 128 GB) to train the student network

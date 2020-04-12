The teacher network is adapted from https://github.com/kuangliu/pytorch-cifar . We used a VGG-11 network. 

Run the svhn_script.sh to train the teacher and student networks and to generate the VHDL code.

In rinc/my_pool_multi_1.py, change p = pool(30) (line 88) to number of cores in your CPU.

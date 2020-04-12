#!/bin/bash

echo "Script for CIFAR PoET-BiN"

echo "Train teacher network - Press ctrl-c to stop training"   

python main.py

python main.py --resume --lr=0.03

python main.py --resume --lr=0.01

echo "Generating binary input dataset and intermediate layer outputs to train decision trees" 

python stitch_6inp.py

echo "Copying generated files" 

cp dt* new_st_class_8/

cp *labels.npy new_st_class_8/

cd new_st_class_8/

echo "Training 80 DTs in parallel"

python my_pool_multi_1.py

echo "Copying predicted outputs of the DTs"

cp predicted* class_40_8/

cp *labels.npy class_40_8/

cd class_40_8/

echo "Re-training the sparse output layer"

python main.py

echo "Generating last layer VHDL"

python gen_last_vhdl.py

echo "Generating RINC modules VHDL"

cd ..

python vhdl_gen_v1.py

echo "Generating Testbench"

python fin_tb.py







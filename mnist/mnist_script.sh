#!/bin/bash

echo "Script for MNIST PoET-BiN"

echo "Train teacher network"   

python main.py

echo "Generating binary input dataset and intermediate layer outputs to train decision trees" 

python stitch_6inp.py

echo "Copying generated files" 

cp dt* classifier/

cp *labels.npy classifier/

cd classifier/

echo "Training 80 DTs in parallel"

python my_pool_multi_1.py

echo "Copying predicted outputs of the DTs"

cp predicted* class_8/

cp *labels.npy class_8/

cd class_8/

echo "Re-training the sparse output layer"

python main.py

echo "Generating last layer VHDL"

python gen_last_vhdl.py

echo "Generating RINC modules VHDL"

cd ..

python vhdl_gen_v1.py

echo "Generating Testbench"

python fin_tb.py







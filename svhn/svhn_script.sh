#!/bin/bash

echo "Script for SVHN PoET-BiN"

echo "Train teacher network - Press ctrl-c to stop training"   

python main.py

python main.py --resume --lr=0.001

python main.py --resume --lr=0.0005

echo "Generating binary input dataset and intermediate layer outputs to train decision trees" 

python stitch_6inp.py

echo "Copying generated files" 

cp dt* rinc/

cp *labels.npy rinc/

cd rinc/

echo "Training 60 DTs in parallel"

python my_pool_multi_1.py

echo "Copying predicted outputs of the DTs"

cp predicted* classifier_36_6/

cp *labels.npy classifier_36_6/

cd classifier_36_6/

echo "Re-training the sparse output layer"

python main.py

echo "Generating last layer VHDL"

python gen_last_vhdl.py

echo "Generating RINC modules VHDL"

cd ..

python vhdl_gen_v1.py

echo "Generating Testbench"

python fin_tb.py







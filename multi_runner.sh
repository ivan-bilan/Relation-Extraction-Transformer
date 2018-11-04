#!/bin/bash

python runner.py --seed 1234 --id aaa_1  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention
python runner.py --seed 1234 --id aaa_2  --save_epoch 10  --num_epoch 200 --diagonal_positional_attention --lr 0.5  --lr_decay 0.8
python runner.py --seed 1234 --id aaa_3  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention --num_layers_encoder 2
python runner.py --seed 1234 --id aaa_4  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention --n_head 6

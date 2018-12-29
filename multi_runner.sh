#!/bin/bash

# you can run multiple experiments sequentially using this file

python runner.py --seed 1111 --id aaa_ff9  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --old_residual
python runner.py --seed 1111 --id aaa_ff10  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --use_layer_norm
python runner.py --seed 1111 --id aaa_ff11  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --no-attn
python runner.py --seed 1111 --id aaa_ff12  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --optim adam --lr 0.001
python runner.py --seed 1111 --id aaa_ff13  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 2
python runner.py --seed 1111 --id aaa_ff14 --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 3
python runner.py --seed 1111 --id aaa_ff15  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 6
python runner.py --seed 1111 --id aaa_ff16  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --no_relative_positions

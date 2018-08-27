#!/bin/bash

# This is an example script of training and running model ensembles.
# train 5 models with different seeds
# python runner.py --seed 1111 --id aaa_ff1  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --lr 0.01
# python runner.py --seed 1111 --id aaa_ff2 --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --lr 0.5
# python runner.py --seed 1111 --id aaa_ff3  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --lr 1.0
# python runner.py --seed 1111 --id aaa_ff4  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --n_head 4
# python runner.py --seed 1111 --id aaa_ff5  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --n_head 2
# python runner.py --seed 1111 --id aaa_ff6  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --n_head 8
# python runner.py --seed 1111 --id aaa_ff7  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --weight_no_rel 0.3
# python runner.py --seed 1111 --id aaa_ff8 --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --weight_rest 0.3
# break
python runner.py --seed 1111 --id aaa_ff9  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --old_residual
python runner.py --seed 1111 --id aaa_ff10  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --use_layer_norm
python runner.py --seed 1111 --id aaa_ff11  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --no-attn
python runner.py --seed 1111 --id aaa_ff12  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --optim adam --lr 0.001
python runner.py --seed 1111 --id aaa_ff13  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 2
python runner.py --seed 1111 --id aaa_ff14 --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 3
python runner.py --seed 1111 --id aaa_ff15  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --num_layers_encoder 6
python runner.py --seed 1111 --id aaa_ff16  --save_epoch 10  --diagonal_positional_attention  --num_epoch 100 --no_relative_positions

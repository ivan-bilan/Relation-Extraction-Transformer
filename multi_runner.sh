#!/bin/bash

# This is an example script of training and running model ensembles.
# train 5 models with different seeds
python runner.py --seed 1234 --id x6  --save_epoch 10  --diagonal_positional_attention --weight_rest 1.0
python runner.py --seed 1234 --id x7  --save_epoch 10  --diagonal_positional_attention --old_residual
python runner.py --seed 1234 --id x8  --save_epoch 10  --diagonal_positional_attention --use_layer_norm
python runner.py --seed 1234 --id x9  --save_epoch 10  --diagonal_positional_attention --no-attn
python runner.py --seed 1234 --id x10  --save_epoch 10  --diagonal_positional_attention --self_att_and_rnn
python runner.py --seed 1234 --id x11  --save_epoch 10  --diagonal_positional_attention --no_relative_positions

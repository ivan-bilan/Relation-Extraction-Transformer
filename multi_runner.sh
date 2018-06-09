#!/bin/bash

# This is an example script of training and running model ensembles.
# train 5 models with different seeds
# python runner.py --seed 1234 --id 12 --diagonal_positional_attention --save_epoch 10 --num_epoch 80 --no-attn
# python runner.py --seed 1234 --id 13 --diagonal_positional_attention --save_epoch 10 --num_epoch 80 --batch_size 100
python runner.py --seed 1234 --id 21  --save_epoch 10 --num_epoch 80  --diagonal_positional_attention
python runner.py --seed 1234 --id 22  --save_epoch 10 --num_epoch 80  --dropout 0.1 --scaled_dropout 0.1
python runner.py --seed 1234 --id 23 --save_epoch 10 --num_epoch 80  --use_layer_norm
python runner.py --seed 1234 --id 24  --save_epoch 10 --num_epoch 80  --batch_size 100
python runner.py --seed 1234 --id 25  --save_epoch 10 --num_epoch 80  --self_att_and_rnn
python runner.py --seed 1234 --id 26  --save_epoch 10 --num_epoch 80  --weight_rest 1.0
python runner.py --seed 1234 --id 27  --save_epoch 10 --num_epoch 80  --word_dropout 0.4
python runner.py --seed 1234 --id 28 --save_epoch 10 --num_epoch 80  --num_layers_encoder 2 --n_head 4
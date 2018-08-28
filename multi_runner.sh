#!/bin/bash

# This is an example script of training and running model ensembles.
# train 5 models with different seeds
# python runner.py --seed 2342234 --id aaa1  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 2365757 --id aaa2 --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 5675677 --id aaa3  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 5756775 --id aaa4  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 7568888 --id aaa5  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 5464456 --id aaa6  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 1234 --id aaa7  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 1111 --id aaa8  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 2222 --id aaa9  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 3333 --id aaa10  --save_epoch 10  --diagonal_positional_attention
# python runner.py --seed 1 --id aaa11  --save_epoch 10  --diagonal_positional_attention
# scripts to test softmax(attn)
python runner.py --seed 1234 --id aaa_bins_4  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention
python runner.py --seed 1234 --id aaa_bins_5  --save_epoch 10  --num_epoch 200 --diagonal_positional_attention --lr 0.5  --lr_decay 0.8
python runner.py --seed 1234 --id aaa_bins_6  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention --num_layers_encoder 2
python runner.py --seed 1234 --id aaa_bins_7  --save_epoch 10  --num_epoch 100 --diagonal_positional_attention --n_head 6

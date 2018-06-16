#!/bin/bash

# This is an example script of training and running model ensembles.
# train 5 models with different seeds
python runner.py --seed 1234 --id xx00 --save_epoch 20
python runner.py --seed 1111 --id xx01 --save_epoch 20
python runner.py --seed 2222 --id xx02 --save_epoch 20
python runner.py --seed 3333 --id xx03 --save_epoch 20
python runner.py --seed 4444 --id xx04 --save_epoch 20

# evaluate on test sets and save prediction files
python eval.py --model_dir saved_models/xx00 --model checkpoint_epoch_60.pt --out saved_models/out/test_x0.pkl
python eval.py --model_dir saved_models/xx01 --model checkpoint_epoch_60.pt --out saved_models/out/test_x1.pkl
python eval.py --model_dir saved_models/xx02 --model checkpoint_epoch_60.pt --out saved_models/out/test_x2.pkl
python eval.py --model_dir saved_models/xx03 --model checkpoint_epoch_60.pt --out saved_models/out/test_x3.pkl
python eval.py --model_dir saved_models/xx04 --model checkpoint_epoch_60.pt --out saved_models/out/test_x4.pkl

# run ensemble
ARGS=""
# for id in 1 2 3 4 5; do
for id in x0 x1 x2 x3 x4; do
    OUT="saved_models/out/test_${id}.pkl"
    ARGS="$ARGS $OUT"
done
python ./ensemble.py --dataset test $ARGS
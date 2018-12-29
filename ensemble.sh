#!/bin/bash

# This is an example script of training and running model ensembles.

python runner.py --seed 11 --id xx00 --save_epoch 20
python runner.py --seed 22 --id xx01 --save_epoch 20
python runner.py --seed 44 --id xx03 --save_epoch 20
python runner.py --seed 55 --id xx04 --save_epoch 20
python runner.py --seed 66 --id xx05 --save_epoch 20

# evaluate on test sets and save prediction files
python eval.py --model_dir saved_models/xx00 --model checkpoint_epoch_60.pt --out saved_models/out/test_x0.pkl --seed 11
python eval.py --model_dir saved_models/xx01 --model checkpoint_epoch_60.pt --out saved_models/out/test_x2.pkl --seed 33
python eval.py --model_dir saved_models/xx03 --model checkpoint_epoch_60.pt --out saved_models/out/test_x3.pkl --seed 44
python eval.py --model_dir saved_models/xx04 --model checkpoint_epoch_60.pt --out saved_models/out/test_x4.pkl --seed 55
python eval.py --model_dir saved_models/xx05 --model checkpoint_epoch_60.pt --out saved_models/out/test_x1.pkl --seed 66

# run ensemble
ARGS=""
for id in x0 x1 x2 x3 x4; do
    OUT="saved_models/out/test_${id}.pkl"
    ARGS="$ARGS $OUT"
done
python ./ensemble.py --dataset test $ARGS
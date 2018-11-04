#!/bin/bash
python eval.py --model_dir saved_models/aaa_1 --model checkpoint_epoch_60.pt
python eval.py --model_dir saved_models/aaa_2 --model checkpoint_epoch_60.pt
python eval.py --model_dir saved_models/aaa_3 --model checkpoint_epoch_60.pt
python eval.py --model_dir saved_models/aaa_4 --model checkpoint_epoch_60.pt
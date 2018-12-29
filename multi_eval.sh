#!/bin/bash

# you can evaluate multiple experiments sequentially using this script

python eval.py --model_dir saved_models/21
python eval.py --model_dir saved_models/22
python eval.py --model_dir saved_models/23
python eval.py --model_dir saved_models/24
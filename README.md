Position-Aware Self-Attention for Relation Extraction
=====================================================

"Relation extraction using deep neural networks and self-attention"

```
The Center for Information and Language Processing (CIS)
Ludwig Maximilian University of Munich
Ivan Bilan
```

https://arxiv.org/abs/1807.03052

## Requirements

- Python 3.6+
- PyTorch 0.4.1+ (including 1.0dev)
- CUDA 9.0+ (including CUDA 10)
- CuDNN 7.005 (up to 7.1)

# How to setup
## 1. Dataset

The TACRED dataset used for evaluation is currently not publicly available. Follow the original authors' GitHub page
for more updates: https://github.com/yuhaozhang/tacred-relation

On this page a sample dataset is available at:
https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred

For this implementation, we use the JSON format of the dataset which can be generated with the JSON generations
script included in the dataset.

## 2. Vocabulary preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.



## Training

Train our final model with:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 00 
--info "Position-aware attention model with self-attention encoder"
```

Use `--topn N` to fine-tune the top N word vectors only. The script will do the preprocessing automatically 
(word dropout, entity masking, etc.).

To train a self-attention encoder model only use:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no-attn --id 01 --info "self-attention model"
```

To combine a self-attention encoder model, LSTM and position-aware layer use:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --self_att_and_rnn --id 01 --info "combined model"
```

To train the LSTM only baseline mode, use:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no_self_att --no-attn --id 01 --info "baseline model"
```

To use absolute positional encodings in self-attention instead of relative ones, use:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no_diagonal_positional_attention --id 01 
--info "no relative pos encodings"
```



Model checkpoints and logs will be saved to `./saved_models/00`.

## Evaluation

Run evaluation on the test set with:
```
python eval.py --model_dir saved_models/00
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model 
checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).
In our evaluation runs, we always evaluate the last epoch checkpoint, namely `--model checkpoint_epoch_60.pt` using:
```
python eval.py --model_dir saved_models/00 --model checkpoint_epoch_60.pt
```

## Ensemble

In order to run the ensembled model use: 
 ```
 bash ensemble.sh
 ```

## Best results

Results on the evaluation set:
```
Precision (micro): 65.386%
   Recall (micro): 68.060%
       F1 (micro): 66.696%
```

Per-relation statistics:
 ```
org:alternate_names                  P:  74.78%  R:  80.75%  F1:  77.65%  #: 213
org:city_of_headquarters             P:  71.59%  R:  76.83%  F1:  74.12%  #: 82
org:country_of_headquarters          P:  55.70%  R:  40.74%  F1:  47.06%  #: 108
org:dissolved                        P: 100.00%  R:   0.00%  F1:   0.00%  #: 2
org:founded                          P:  84.21%  R:  86.49%  F1:  85.33%  #: 37
org:founded_by                       P:  72.22%  R:  38.24%  F1:  50.00%  #: 68
org:member_of                        P: 100.00%  R:   0.00%  F1:   0.00%  #: 18
org:members                          P:   0.00%  R:   0.00%  F1:   0.00%  #: 31
org:number_of_employees/members      P:  65.22%  R:  78.95%  F1:  71.43%  #: 19
org:parents                          P:  40.00%  R:  19.35%  F1:  26.09%  #: 62
org:political/religious_affiliation  P:  25.81%  R:  80.00%  F1:  39.02%  #: 10
org:shareholders                     P:  75.00%  R:  23.08%  F1:  35.29%  #: 13
org:stateorprovince_of_headquarters  P:  64.18%  R:  84.31%  F1:  72.88%  #: 51
org:subsidiaries                     P:  55.17%  R:  36.36%  F1:  43.84%  #: 44
org:top_members/employees            P:  66.44%  R:  84.68%  F1:  74.46%  #: 346
org:website                          P:  53.33%  R:  92.31%  F1:  67.61%  #: 26
per:age                              P:  78.06%  R:  92.50%  F1:  84.67%  #: 200
per:alternate_names                  P:   0.00%  R:   0.00%  F1:   0.00%  #: 11
per:cause_of_death                   P:  63.64%  R:  40.38%  F1:  49.41%  #: 52
per:charges                          P:  66.91%  R:  90.29%  F1:  76.86%  #: 103
per:children                         P:  38.30%  R:  48.65%  F1:  42.86%  #: 37
per:cities_of_residence              P:  52.91%  R:  62.43%  F1:  57.28%  #: 189
per:city_of_birth                    P:  50.00%  R:  20.00%  F1:  28.57%  #: 5
per:city_of_death                    P: 100.00%  R:  21.43%  F1:  35.29%  #: 28
per:countries_of_residence           P:  50.00%  R:  55.41%  F1:  52.56%  #: 148
per:country_of_birth                 P: 100.00%  R:   0.00%  F1:   0.00%  #: 5
per:country_of_death                 P: 100.00%  R:   0.00%  F1:   0.00%  #: 9
per:date_of_birth                    P:  77.78%  R:  77.78%  F1:  77.78%  #: 9
per:date_of_death                    P:  62.16%  R:  42.59%  F1:  50.55%  #: 54
per:employee_of                      P:  64.34%  R:  69.70%  F1:  66.91%  #: 264
per:origin                           P:  68.81%  R:  56.82%  F1:  62.24%  #: 132
per:other_family                     P:  59.09%  R:  43.33%  F1:  50.00%  #: 60
per:parents                          P:  58.82%  R:  56.82%  F1:  57.80%  #: 88
per:religion                         P:  44.16%  R:  72.34%  F1:  54.84%  #: 47
per:schools_attended                 P:  64.29%  R:  60.00%  F1:  62.07%  #: 30
per:siblings                         P:  61.29%  R:  69.09%  F1:  64.96%  #: 55
per:spouse                           P:  56.58%  R:  65.15%  F1:  60.56%  #: 66
per:stateorprovince_of_birth         P:  40.00%  R:  50.00%  F1:  44.44%  #: 8
per:stateorprovince_of_death         P:  80.00%  R:  28.57%  F1:  42.11%  #: 14
per:stateorprovinces_of_residence    P:  65.28%  R:  58.02%  F1:  61.44%  #: 81
per:title                            P:  77.13%  R:  87.00%  F1:  81.77%  #: 500
 ```
 
 ## Acknowledgement
 
 The self-attention implementation in this project is mostly taken from (all modifications are explained in the paper linked above): 
 [Attention is all you need: A Pytorch Implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch) 
 (Related code licensed under [MIT License](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/LICENSE)).
 
 The original TACRED implementation is used as a base of this implementation (all modifications are explained in the 
 paper linked above): [Position-aware Attention RNN Model for Relation Extraction](https://github.com/yuhaozhang/tacred-relation)
  (Related code licensed under [Apache License, Version 2.0](https://github.com/yuhaozhang/tacred-relation/blob/master/LICENSE)).
  
## License

All original code in this project is licensed under the Apache License, Version 2.0. See the included LICENSE file.
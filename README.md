Position-Aware Self-Attention for Relation Extraction
=====================================================

WORK IN PROGRESS! Ideas, bug-fixes and constructive criticism are all welcome.

This project is the result of my Master's Thesis (supervised by
 [Dr. Benjamin Roth](http://www.cis.uni-muenchen.de/personen/mitarbeiter/beroth/index.html)):
```
"Relation extraction using deep neural networks and self-attention"
The Center for Information and Language Processing (CIS)
Ludwig Maximilian University of Munich
Ivan Bilan
```

The pre-print is available on arXiv (in collaboration with
 [Dr. Benjamin Roth](http://www.cis.uni-muenchen.de/personen/mitarbeiter/beroth/index.html)):
 
https://arxiv.org/abs/1807.03052

Related presentation from PyData Berlin 2018:

[Understanding and Applying Self-Attention for NLP - Ivan Bilan](https://www.youtube.com/watch?v=OYygPG4d9H0)

## Requirements

- Python 3.6+
- PyTorch 0.4.1+ (including 1.0dev)
- CUDA 9.0+ (including CUDA 10 if you build PyTorch from source, not compatible with CuDNN <7.2)
- CuDNN 7.005 (up to 7.1)

# How to setup
## 1. Python Environment

To automatically create a conda environment (using Anaconda3) with Python 3.7 and Pytorch 1.0dev, run the following command:
```bash
make build_venv
```

Note: you have to have CUDA installed already before creating the environment.

## 2. Dataset

The TACRED dataset used for evaluation is currently not publicly available. Follow the original authors' GitHub page
for more updates: https://github.com/yuhaozhang/tacred-relation

On this page a sample dataset is available at:
https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred

For this implementation, we use the JSON format of the dataset which can be generated with the JSON generations
script included in the dataset.

## 3. Vocabulary preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.


# Project Usage

## 1. Training

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

## 2. Evaluation

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

## 3. Ensemble Training

In order to run the ensembled model use: 
 ```
 bash ensemble.sh
 ```

# Best results

Results comparison on evaluation set (single model):

| Evaluation Metric         | Our approach           | Zhang et al. 2017  |
| ------------- |:-------------:| -----:|
| Precision (micro)      | 65.4% | **65.7%** |
| Recall (micro)      | **68.0%**      |   64.5% |
| F1 (micro) | **66.7%**      |    65.1% |

Per-relation statistics (single model):
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
 
If you can't replicate the results on the master branch, run `pytorch_0_4_1_wip_version_2` using PyTorch 0.4.1.

## Overview of Available Hyperparameters

| **General Hyperparameters**      | | |
|-------------:| -----:| -----:|
| **Argument Name**           | **Default Value**  | **Description** |
| `--emb_dim`     |   `300` | Word embeddings dimension size | 
| `--word_dropout`     |   `0.06` | The rate at which we randomly set a word to `UNK` | 
| `--lower / --no-lower`     |   `True` | Lowercase all words | 
| `--weight_no_rel`     |   `1.0` | Weight for `no_relation` class | 
| `--weight_rest`     |   `1.0` | Weight for other classes but `no_relation`  | 
| `--lr`     |   `0.1` | Learning rate (Applies to SGD and Adagrad only)  | 
| `--lr_decay`     |   `0.9` | Learning rate decay  | 
| `--decay_epoch`     |   `15` | Start learning rate decay from given epoch  | 
| `--max_grad_norm`     |   `1.0` | Gradient clipping value  | 
| `--optim`     |   `sgd` | Optimizer, available options: `sgd, asgd, adagrad, adam, nadam, noopt_adam, openai_adam, adamax`  | 
| `--num_epoch`     |   `70` | Number of epochs | 
| `--batch_size`     |   `50` | Batch size  | 
| `--topn`     |   `1e10` | Only fine-tune top N embeddings  | 
| `--log_step`     |   `400` | Print log every k steps  | 
| `--log`     |   `logs.txt` | Write training log to specified file  | 
| `--save_epoch`     |   `1` | Save model checkpoints every k epochs  | 
| `--save_dir`     |   `./saved_models` | Root dir for saving models  | 
| **Position-aware Attention Layer**      | | |
| `--ner_dim`     |   `30` | NER embedding dimension  |
| `--pos_dim`     |   `30` | POS embedding dimension  |
| `--pe_dim`     |   `30` | Position encoding dimension in the attention layer  |
| `--attn_dim`     |   `200` | Attention size in the attention layer |
| `--query_size_attn`     |   `360` | Embedding for query size in the positional attention  |
| `--attn / --no-attn`     |   `True` | Use the position-aware attention layer  |
| **Position-aware Attention LSTM Layer**      | | |
| `--hidden_dim`     |   `360` | LSTM hidden state size  |
| `--num_layers`     |   `2` | Number of LSTM layers |
| `--lstm_dropout`     |   `0.5` | LSTM dropout rate |
| `--self_att_and_rnn / --no_self_att_and_rnn`     |   `False` | Use LSTM layer with the Self-attention layer |
| **Self-attention**      | | |
| `--num_layers_encoder`     |   `1` | Number of self-attention encoders |
| `--n_head`     |   `3` | Number of self-attention heads |
| `--dropout`     |   `0.4` | Input and attention dropout rate |
| `--hidden_self`     |   `130` | Encoder layer width |
| `--scaled_dropout`     |   `0.1` | `ScaledDotProduct` Attention dropout |
| `--temper_value`     |   `0.5` | Temper value for `ScaledDotProduct` Attention |
| `--use_batch_norm`     |   `True` | Use BatchNorm in Self-attention |
| `--use_layer_norm`     |   `False` | Use LayerNorm in Self-attention |
| `--new_residual`     |   `True` | Use a different residual connection structure than in the original Self-attention |
| `--old_residual`     |   `False` | Use the original residual connections in Self-attention |
| `--obj_sub_pos`     |   `True` | In self-attention add object and subject positional vectors |
| `--relative_positions / --no_relative_positions`     |   `True` | Bin the relative positional encodings |
| `--diagonal_positional_attention / --no_diagonal_positional_attention`     |   `True` | Use relative positional encodings as described in our paper |
| `--self-attn / --no_self_att`     |   `True` | Use the Self-attention encoder |
| **Lemmatize input**      | | |
| `--use_lemmas / no_lemmas`     |   `False` | Instead of raw text, use spaCy to lemmatize the sentences |
| `--preload_lemmas / --no_preload_lemmas`     |   `False` | Preload lemmatized input as pickles |


## Attention Example
 
Sample Sentence from TACRED: 
 
 *They cited the case of __Agency for International Development__ (OBJECT) subcontractor __Alan Gross__ (SUBJECT),
 who was working in Cuba on a tourist visa and possessed satellite communications equipment, who has been held in a 
 maximum security prison since his arrest Dec 3.*
 
 Attention distribution for the preposition **of** in the sentence above:
![Attention Distribution](./experiments/attention_plot.png/?raw=true "Attention Distribution Example")
 
 
 # Acknowledgement
 
 The self-attention implementation in this project is mostly taken from (all modifications are explained in the paper linked above): 
 [Attention is all you need: A Pytorch Implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch) 
 (Related code licensed under [MIT License](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/LICENSE)).
 
 The original TACRED implementation is used as a base of this implementation (all modifications are explained in the 
 paper linked above): [Position-aware Attention RNN Model for Relation Extraction](https://github.com/yuhaozhang/tacred-relation)
  (Related code licensed under [Apache License, Version 2.0](https://github.com/yuhaozhang/tacred-relation/blob/master/LICENSE)).
  
# License

All original code in this project is licensed under the Apache License, Version 2.0. See the included LICENSE file.

# TODOs

* Improve and document attention visualization process
* Add weighting functions as hyperparameter
* Add tests
* Currently the project is hard-coded to work on a GPU, add CPU support
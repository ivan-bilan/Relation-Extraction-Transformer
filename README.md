Self-Attention for Relation Extraction
=========================

"Relation extraction using deep neural networks and self-attention"

```
The Center for Information and Language Processing (CIS)
Ludwig Maximilian University of Munich
Ivan Bilan
```

## Requirements

- Python 3.6.2
- PyTorch 0.4
- CUDA 9.1
- CuDNN 7.005

## Preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Dataset

The TACRED dataset used for evaluation is currently not publicly available. Follow the original authors' GitHub page
for more updates: https://github.com/yuhaozhang/tacred-relation. On this page a sample dataset is available at:
https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred

For this implementation, we use the JSON format of the dataset which can be generated with the JSON generations
script included in the dataset.

## Training

Train our final model with:
```
python runner.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 00 
--info "Position-aware attention model with self-attention encoder"
```

Use `--topn N` to finetune the top N word vectors only. The script will do the preprocessing automatically 
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
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model 
checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).

## Ensemble

In order to run the ensembled model use: 
 ```
 bash ensemble.sh
 ```
 
 ## Limitations 
 
 Currently there is problem with running the model on certain hardware/software constellation due to 
 the beta state of the PyTorch. I was tested and works properly with CUDA 9.1 and CuDNN 7.005. If there are
 any issues try running the model without the relative positional embeddings by including the flag 
 `--no_diagonal_positional_attention`. Also follow the GitHub page of the project for regular updated when the
 open source release is available at https://github.com/ivan-bilan.
 
 

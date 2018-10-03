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
- PyTorch 0.4.0+
- CUDA 9.0+ (including CUDA 10)
- CuDNN 7.005 (up to 7.1)

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

 

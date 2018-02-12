Self-Attention for Relation Extraction
=========================

Augmenting attention mechanisms for Relation Extraction.

The Center for Information and Language Processing (CIS)

Ludwig Maximilian University of Munich

Ivan Bilan


## Requirements

- Python 3.6.2
- PyTorch 0.3

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

## Training

Train a position-aware attention RNN model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 00 --info "Position-aware attention model"
```

Use `--topn N` to finetune the top N word vectors only. The script will do the preprocessing automatically (word dropout, entity masking, etc.).

Train an LSTM model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no-attn --id 01 --info "LSTM model"
```

Model checkpoints and logs will be saved to `./saved_models/00`.

## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).

## Ensemble

Please see the example script `ensemble.sh`.

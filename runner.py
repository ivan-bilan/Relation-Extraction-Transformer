"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab

import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=360, help='RNN hidden state size.')            # 200 original
parser.add_argument('--num_layers', type=int, default=2, help='Num of lstm layers.')
parser.add_argument('--num_layers_encoder', type=int, default=3, help='Num of self-attention encoders.')
parser.add_argument('--dropout', type=float, default=0.6, help='Input and attn dropout rate.')        # 0.5 original
parser.add_argument('--scaled_dropout', type=float, default=0.1, help='Input and scaled dropout rate.')        # 0.1 original
parser.add_argument('--word_dropout', type=float, default=0.04,                                      # 0.04
                    help='The rate at which randomly set a word to UNK.'
                   )
parser.add_argument('--lstm_dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.',
                   default=True)

parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--weight_no_rel', type=float, default=2.0, help='Weight for no_relation class.')
parser.add_argument('--weight_rest', type=float, default=1.0, help='Weight for other classes.')

parser.add_argument(
    '--self-attn', dest='self_att', action='store_true',
    help='Use self-attention layer instead of LSTM.', default=True
)

parser.add_argument('--obj_sub_pos', dest='obj_sub_pos', action='store_true',
    help='In self-attention add obj/subg positional vectors.', default=True)
parser.add_argument('--use_batch_norm', dest='use_batch_norm', action='store_true',
    help='BatchNorm if True, else LayerNorm in self-attention.', default=True)
parser.add_argument('--relative_positions', dest='relative_positions', action='store_true',
    help='Use relative positions for subj/obj positional vectors.', default=True)
parser.add_argument('--new_residual', dest='new_residual', action='store_true',
    help='Use a different residual connection than in usual self-attention.', default=True)

parser.add_argument('--n_head', type=int, default=1, help='Number of self-attention heads.')
parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.', default="true")
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)

parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')                    # 200 original
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--lr', type=float, default=0.3, help='Applies to SGD and Adagrad.')            # lr 1.0 orig
parser.add_argument('--lr_decay', type=float, default=0.95)                                          # lr_decay 0.9 original
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')       # sgd original
parser.add_argument('--num_epoch', type=int, default=200)                                           # epochs 30 original
parser.add_argument('--batch_size', type=int, default=50)                                           # batch size 50 original
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')

# info for model saving
parser.add_argument('--log_step', type=int, default=400, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=10, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

parser.add_argument(
    '--id', type=str,
    default='38_self_attention_dropout',                                 # change model folder output before running
    help='Model ID under which to save models.'
   )

parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# improves speed of cuda somehow, set to False by default due to memory usage
torch.backends.cudnn.fastest=True
torch.backends.cudnn.benchmark=True

# start training
for epoch in range(1, opt['num_epoch']+1):

    print(
        "Current params: "+ " heads-"+ str(opt["n_head"]) + " enc_layers-" + str(opt["num_layers_encoder"]),
        " drop-"+ str(opt["dropout"]) + " scaled_drop-" + str(opt["scaled_dropout"]) + " lr-"+ str(opt["lr"]),
        " lr_decay-"+ str(opt["lr_decay"]) + " grad_norm-"+ str(opt["max_grad_norm"])
    )
    print(
        " weight_no_rel-"+ str(opt["weight_no_rel"]) +
        " weight_rest-"+ str(opt["weight_rest"]) + " attn-"+ str(opt["attn"]) +" attn_dim-"+ str(opt["attn_dim"]),
        " obj_sub_pos-"+ str(opt["obj_sub_pos"]) + " new_residual-"+str(opt["new_residual"]),
        " use_batch_norm-"+str(opt["use_batch_norm"]) + " relative_positions-"+str(opt["relative_positions"])
    )

    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(
                format_str.format(datetime.now(), global_step, max_steps, epoch,
                opt['num_epoch'], loss, duration, current_lr)
            )
        # do garbage collection, as per https://discuss.pytorch.org/t/best-practices-for-maximum-gpu-utilization/13863/6
        del loss

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = model.predict(batch)
        predictions += preds
        dev_loss += loss
        
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print(
        "epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1)
        )
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

    # save the model during every epoch, if fscore is the best, move it to best_model.pkl
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    # delete single checkpoints based on the save_epoch int
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and opt['optim'] in ['sgd', 'adagrad', 'adam']:
        # don't go lower than 0.01 lr
        if current_lr >= 0.01:
            current_lr *= opt['lr_decay']
            model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

# !!!!!!!! change the model output folder !!!!!!!!!!!

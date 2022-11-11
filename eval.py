"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from global_random_seed import RANDOM_SEED

from data.loader import DataLoader
from model.relation_model import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, help='Directory of the model.',
    default="saved_models/tmp5/"
)
# parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str,
                    default="saved_models/out/test_6.pkl",
                    help="Save model predictions to this dir."
)

parser.add_argument('--seed', type=int, default=RANDOM_SEED)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

args = parser.parse_args()

with open('global_random_seed.py', 'w') as the_file:
    the_file.write('RANDOM_SEED = '+str(args.seed))

# set top level random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)

if args.cpu:
    args.cuda = False
elif args.cuda:
    # set random seed for cuda as well
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
# TODO: are we using dropout in testing??
# opt["dropout"] = 0.0
# opt["scaled_dropout"] = 0.0

model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []

with torch.no_grad():
    for i, b in enumerate(batch):
        preds, probs, _ = model.predict(b)
        predictions += preds
        all_probs += probs

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

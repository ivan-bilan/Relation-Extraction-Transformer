import re
import json
import math
import pickle
import random
import torch
import numpy as np

from tqdm import tqdm

from utils import constant, helper, vocab
from global_random_seed import RANDOM_SEED
from utils.extract_lemmas import extract_lemmas

"""
Data loader for TACRED json files.
"""

PAD = 0
ABS_MAX_LEN = 96

# make everything reproducible
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):

        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        if opt["use_lemmas"] and not opt["preload_lemmas"]:
            import spacy
            # load the spacy model
            self.nlp = spacy.load('en_core_web_lg')

        # read the json file with data
        with open(filename) as infile:
            data = json.load(infile)

        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """

        processed = list()
        # max_sequence_length = 0 # it's 96 now

        lemmatized_tokens = list()
        if opt["preload_lemmas"] and opt["use_lemmas"]:
            if len(data) == 68124 and opt["preload_lemmas"]:
                with open('dataset/spacy_lemmas/train_lemmatized.pkl', 'rb') as f:
                    lemmatized_tokens = pickle.load(f)
            elif len(data) == 22631 and opt["preload_lemmas"]:
                with open('dataset/spacy_lemmas/dev_lemmatized.pkl', 'rb') as f:
                    lemmatized_tokens = pickle.load(f)
            elif len(data) == 15509 and opt["preload_lemmas"]:
                with open('dataset/spacy_lemmas/test_lemmatized.pkl', 'rb') as f:
                    lemmatized_tokens = pickle.load(f)
            print("loading lemmatized tokens...")

        for i, d in enumerate(tqdm(data)):

            tokens = d['token']
            if opt["use_lemmas"] and not opt["preload_lemmas"]:
                tokens = extract_lemmas(self.nlp, tokens, i)
                lemmatized_tokens.append(tokens)
            elif opt["use_lemmas"] and opt["preload_lemmas"]:
                tokens = lemmatized_tokens[i]

            # TODO: get max sequence length (within batch?)
            # if max_sequence_length <= len(d['token']):
            #    max_sequence_length = len(d['token'])

            # lowercase all tokens
            if opt['lower']:
                # print("LOWERIN")
                tokens = [t.lower() for t in tokens]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)

            tokens = map_to_ids(tokens, vocab.word2id)

            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)

            # create word positional vector for self-attention
            inst_position = list([pos_i + 1 if w_i != PAD else 0 for pos_i, w_i in enumerate(tokens)])
            # print("inst_position", inst_position)

            # double the amount of positional embeddings for the diagonal positional attention
            relative_positions = self.bin_positions(get_position_modified(l - 1, l - 1, l * 2 - 1))

            # position relative to Subject and Object are calculated here
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)

            # pass relative positional vectors
            if opt["relative_positions"]:
                # print(subj_positions)
                # do binning for subject positions
                # subj_positions_orig = subj_positions

                # TODO: select proper function to do this
                subj_positions = self.bin_positions(subj_positions)

                # subj_positions = self.bin_positions(subj_positions, 2)
                # do binning for object positions
                # print(obj_positions)

                # obj_positions_orig = obj_positions
                obj_positions = self.bin_positions(obj_positions)

                # obj_positions = self.bin_positions(obj_positions, 2)
                # print(obj_positions)

            # one-hot encoding for relation classes
            relation = constant.LABEL_TO_ID[d['relation']]

            # print("inst_position", inst_position, type(inst_position), inst_position)
            # print("subj_positions", subj_positions, type(subj_positions), subj_positions)

            # return vector of the whole partitioned data
            processed += [
                (tokens, pos, ner, deprel, subj_positions, obj_positions, relative_positions,
                 inst_position, relation)
            ]

        # pickle spacy lemmatized text
        if len(data) == 68124 and opt["use_lemmas"] and not opt["preload_lemmas"]:
            print("saving to pickle...")
            with open('dataset/spacy_lemmas/train_lemmatized.pkl', 'wb') as f:
                pickle.dump(lemmatized_tokens, f)
        elif len(data) == 22631 and opt["use_lemmas"] and not opt["preload_lemmas"]:
            with open('dataset/spacy_lemmas/dev_lemmatized.pkl', 'wb') as f:
                pickle.dump(lemmatized_tokens, f)
        elif len(data) == 15509 and opt["use_lemmas"] and not opt["preload_lemmas"]:
            with open('dataset/spacy_lemmas/test_lemmatized.pkl', 'wb') as f:
                pickle.dump(lemmatized_tokens, f)

        return processed

    def bin_positions(self, positions_list):
        """
        Recalculate the word positions by binning them:
        e.g. input = [-3 -2 -1  0  1  2  3  4  5  6  7]
              --> output=[-2 -2 -1  0  1  2  2  3  3  3  3]

        :param positions_list: list of word positions relative to the query or object
        :return: new positions
        """

        a = np.array(positions_list)
        a[a > 0] = np.floor(np.log2(a[a > 0])) + 1
        a[a < 0] = -np.floor(np.log2(-a[a < 0])) - 1
        return a.tolist()

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """

        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 9

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            # TODO: experiment with word dropouts!
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # TODO: get rid of using indexing to rely on the batch item types

        # get_long_tensor creates a matrix out of list of lists
        # convert to tensors
        words = get_long_tensor(words, batch_size)  # matrix of tokens
        pos = get_long_tensor(batch[1], batch_size)  # matrix of part of speech embeddings
        ner = get_long_tensor(batch[2], batch_size)  # matrix for NER embeddings
        deprel = get_long_tensor(batch[3], batch_size)  # stanford dependency parser stuff... not sure

        subj_positions = get_long_tensor(batch[4], batch_size)  # matrix of positional lists relative to subject
        obj_positions = get_long_tensor(batch[5], batch_size)  # matrix of positional lists relative to object

        # do padding here, it will get the longest sequence and pad the rest
        obj_positions_single = get_long_tensor(batch[6], batch_size)  # matrix, positional ids for all words in sentence

        src_pos = get_long_tensor(batch[7], batch_size)  # matrix, positional ids for all words in sentence

        # new masks with positional padding
        masks = torch.eq(words, 0)  # should we also do +src_pos?
        rels = torch.LongTensor(batch[8])  # list of relation labels for this batch

        return (words, masks, pos, ner, deprel, subj_positions, obj_positions, obj_positions_single, src_pos, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    # print(start_idx, end_idx, length)
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))


def get_position_modified(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    # print(start_idx, end_idx, length)
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))


def get_long_tensor(tokens_list, batch_size):
    """
    Convert list of list of tokens to a padded LongTensor.
    Also perform padding here.
    """

    token_len = max(len(x) for x in tokens_list)

    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)

    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """
    Sort all fields by descending order of lens, and return the original indices.
    """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """
    Randomly dropout tokens (IDs) and replace them with <UNK> tokens.
    """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]

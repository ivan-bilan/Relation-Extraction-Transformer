"""
Data loader for TACRED json files.
"""

import json
import math
import random
import torch
import numpy as np

from utils import constant, helper, vocab


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):

        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as infile:
            data = json.load(infile)

        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """

        processed = []
        # max_sequence_length = 0 # it's 96 now

        for d in data:
            tokens = d['token']

            # get max sequence length
            # if max_sequence_length <= len(d['token']):
            #    max_sequence_length = len(d['token'])

            # lowercase all tokens
            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)

            # position relative to Subject and Object are calculated here
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            # print(subj_positions)
            # do binning for subject positions
            subj_positions = self.relativate_word_positions(subj_positions)
            # subj_positions = self.bin_positions(subj_positions, 2)

            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            # do binning for object positions
            # print(obj_positions)
            obj_positions = self.relativate_word_positions(obj_positions)
            # obj_positions = self.bin_positions(obj_positions, 2)
            # print(obj_positions)

            relation = constant.LABEL_TO_ID[d['relation']]
            processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]

        return processed

    def relativate_word_positions(self, positions_list):
        """

        Recalculate the word positions by decreasing their relativeness based on the distance to
        query or object:
        e.g. input=[0,1,2,3,4,5,5,6] --> output=[0,1,2,3,3,4,4,4]

        :param positions_list: list of word positions relative to the query or object
        :return: new positions
        """
        new_list = [math.ceil(math.log(abs(x)+1, 2)) for x in positions_list]
        new_list_final = list()

        # reverse positives
        for index, element in enumerate(new_list):
            if element == 0:
                # new_list_final.append(element)
                new_list_final.extend(new_list[index:])
                break
            else:
                new_element = -element
                new_list_final.append(new_element)
        if len(positions_list) != len(new_list_final):
            print("Error in positional embeddings!")
        return new_list_final

    def bin_positions(self, positions_list, width):
        """

        Recalculate the word positions binning them given a binning distance:
        e.g. input=[-4,-3,-3,-2,-2,-1,-1,0,0,1,1,2,2,3,3,4] and window=3
              --> output=[-3,-2,-2,-2,-1,-1,-1,0,0,1,1,1,2,2,2,3]

        :param positions_list: list of word positions relative to the query or object
        :param width: width of the bin window
        :return: new positions
        """
        a = np.array(positions_list)
        a[a>0] = (a[a>0]+(width-1))//width
        a[a<0] = (a[a<0])//width
        return a.tolist()

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        # return 50
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
        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        
        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])

        return (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions_original(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """

    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]

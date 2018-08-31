''' Define the Transformer model '''

import math
import copy
import torch
import torch.nn.init as init
import torch.nn as nn
import numpy as np
from .Constants import *
from torch.autograd import Variable
from .Modules import BottleLinear as Linear
from .Layers import EncoderLayer

from global_random_seed import RANDOM_SEED

# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=96):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionalEncodingLookup(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, max_len=96):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


def position_encoding_init(n_position, d_pos_vec):
    """
    Init the sinusoid position encoding table

    :param n_position:
    :param d_pos_vec:
    :return:
    """
    
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    """
    Indicate the padding-related part to mask
    :param seq_q: 
    :param seq_k: 
    :return:
    """

    assert seq_q.dim() == 2 and seq_k.dim() == 2

    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    # print(seq_k)
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)    # b x 1 x sk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # b x sq x sk
    # print(pad_attn_mask)
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """
    Get an attention mask to avoid using the subsequent info.
    :param seq:
    :return:
    """
    assert seq.dim() == 2

    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.to("cuda")

    return subsequent_mask


class Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=3, n_head=1, d_k=360, d_v=360,
            d_word_vec=360, d_model=360, d_inner_hid=720, dropout=0.1, scaled_dropout=0.1, obj_sub_pos=False,
            use_batch_norm=True, residual_bool=False, diagonal_positional_attention=False, relative_positions=False,
            temper_value=0.5
    ):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        # new diagonal positional encodings
        self.diagonal_positional_attention = diagonal_positional_attention
        self.relative_positions = relative_positions

        # decide whether to add subject and object positional vectors to the normal positional vectors
        self.obj_sub_pos = obj_sub_pos

        # make sure all dimensions are correct, based on the paper
        assert d_word_vec == d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        if obj_sub_pos and not self.diagonal_positional_attention:
            # TODO: do we need to learn separate encodings here???
            self.position_enc2 = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
            self.position_enc2.weight.data = position_encoding_init(n_position, d_word_vec)

            self.position_enc3 = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
            self.position_enc3.weight.data = position_encoding_init(n_position, d_word_vec)

            self.positions_enc4 = PositionalEncoding(d_model, 0.1, 96)

        elif self.diagonal_positional_attention:
            # needs a positional matrix double the size of embeddings

            # other
            # self.position_dpa = nn.Parameter(torch.FloatTensor((n_position*2)-1, d_word_vec//n_head).cuda())
            # position_encoding_init((n_position*2)-1, d_word_vec//n_head)

            # working

            self.position_enc2 = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
            # TODO: is it better to learn new encodings here?
            self.position_enc2.weight.data = position_encoding_init(n_position, d_word_vec)
            self.position_enc2.weight.requires_grad = True

            self.position_enc3 = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
            # TODO: is it better to learn new encodings here?
            self.position_enc3.weight.data = position_encoding_init(n_position, d_word_vec)
            self.position_enc3.weight.requires_grad = True

            # TODO: try n_pos, n_pos*2-1 #BR
            self.position_dpa = nn.Embedding((n_position*2)-1, d_word_vec//n_head, padding_idx=PAD)

            # TODO: investigate if we need pos encoding here as well
            # self.position_dpa.weight.data = position_encoding_init((n_position*2)-1, d_word_vec//n_head)
            # init.kaiming_normal_(self.position_dpa)

            # make sure embeddings are trainable for dpa
            self.position_dpa.weight.requires_grad = True

            # print(self.position_dpa)
            # self.position_dpa2 = PositionalEncodingLookup(d_word_vec//n_head, (n_position*2)-1)

        # this is for self-learned embeddings
        # in the original paper they don't use pre-trained embeddings
        # since we use glove, we skip this step
        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)

        # use deep copy based on:
        # http://nlp.seas.harvard.edu/2018/04/03/attention.html
        self.layer_stack = nn.ModuleList([
            copy.deepcopy(EncoderLayer(
                d_model=d_model,
                d_inner_hid=d_inner_hid,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                scaled_dropout=scaled_dropout,
                use_batch_norm=use_batch_norm,
                residual_bool=residual_bool,
                temper_value=temper_value
            ))
            for _ in range(n_layers)])

    def forward(self, enc_non_embedded, src_seq, src_pos, pe_features):

        # original use: https://github.com/jadore801120/attention-is-all-you-need-pytorch

        # Word embedding look up, already done in rnn.py
        # in the original paper they don't use pre-trained embeddings
        # since we use glove, we skip this step
        # enc_input = self.src_word_emb(src_seq)

        # TODO: try adding vectors (word vec + pos vec) instead of just appending them
        # TODO: try experimenting with character-based embeddings?

        # add positional encoding to the initial input, add emd_vec+pos_vec value by value
        # originally we used the positional vector of the sentence from 0 to n+1
        # src_seq += self.position_enc(src_pos)

        position_dpa = None

        # decide whether to add subject and object positional vectors to the normal positional vectors
        if self.obj_sub_pos and not self.diagonal_positional_attention:  # this is missing!!!

            # original 64f score
            # src_seq = src_seq + self.position_enc(src_pos)
            # + self.position_enc2(pe_features[1]) + self.position_enc3(pe_features[0])

            if self.relative_positions:
                # add object positions only
                src_seq = src_seq + self.position_enc2(pe_features[1]) # + self.position_enc3(pe_features[0])
            else:
                # TODO
                # this is a fallback, for some reason non-relative encoding doesn't work for obj/subj positions
                src_seq += self.position_enc(src_pos)  # src_pos

            # new
            # print(pe_features[1])
            # src_seq = self.positions_enc4.forward(src_seq) # self.position_enc2(pe_features[1])  # src_seq +

        elif self.diagonal_positional_attention:

            # first add the obj/subj embeddings to the word embeddings
            # TODO: try all variants here!, also without any obj/subj encodings

            src_seq = src_seq + self.position_enc2(pe_features[1])  # + self.position_enc3(pe_features[0])
            # src_seq += self.position_enc2(pe_features[0])

            verbose_sizes = False

            if verbose_sizes:
                print("src_seq.size():", src_seq.size())
                # TODO: try obj/subj positions
                print("using diagonal positional encodings 0")
                print(pe_features[2])
                print(pe_features[2].size())

            # now we take the modified positional word encodings
            position_dpa = self.position_dpa(pe_features[2])  # src_pos of size 2n

            # position_dpa = self.position_dpa2.forward(pe_features[1])  # src_pos

            if verbose_sizes:
                print("position_dpa.size():", position_dpa.size())

        else:
            src_seq += self.position_enc(src_pos)

        enc_slf_attns = list()
        enc_output = src_seq

        # add masking for attention
        enc_slf_attn_mask = get_attn_padding_mask(enc_non_embedded, enc_non_embedded)

        # iterate over encoder layers
        for enc_layer in self.layer_stack:

            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                slf_attn_mask=enc_slf_attn_mask,
                position_dpa=position_dpa,
                sentence_words=enc_non_embedded
            )

        enc_slf_attns += [enc_slf_attn]
        return enc_output, enc_slf_attns

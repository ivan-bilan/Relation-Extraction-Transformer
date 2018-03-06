''' Define the Layers '''
import torch
import numpy as np
import torch.nn as nn
from .Constants import *
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


# TODO: the original encoder has 2 layers! now we are using just one
class EncoderLayer(nn.Module):
    """
    Compose with two layers
    """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # the longest sentence
        n_position = 96 + 1

        # don't really need embeddings here,
        # this was done so that they can also be trained if needed?
        # not sure, follow this discussion: https://github.com/jadore801120/attention-is-all-you-need-pytorch/issues/33
        # self.position_enc = nn.Embedding(n_position, d_model, padding_idx=PAD)  # here should be d_model = vec_size
        # self.position_enc.weight.data = position_encoding_init(n_position, d_model)

        # attention heads
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )

        # feed forward part
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):  # enc_non_embedded, enc_input, enc_pos,

        # here q, k, w are all the same at input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask
        )

        # do feed forward
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

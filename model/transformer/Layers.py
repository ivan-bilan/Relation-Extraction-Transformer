''' Define the Layers '''
import torch
import numpy as np
import torch.nn as nn
from .Constants import *
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Compose with two layers
    """

    def __init__(
            self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, scaled_dropout=0.1,
            use_batch_norm=True, residual_bool=False
    ):

        super(EncoderLayer, self).__init__()

        # check what implementation of residual to use
        self.residual_bool = residual_bool

        # attention heads
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, scaled_dropout=scaled_dropout,
            use_batch_norm=use_batch_norm, residual_bool=residual_bool
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
        if self.residual_bool:  # use new residual implementation
            enc_output = self.pos_ffn(enc_output, enc_input)
        else:  # typical self-attention representation
            enc_output = self.pos_ffn(enc_output, None)

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

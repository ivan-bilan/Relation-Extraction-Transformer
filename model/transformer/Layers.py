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
            use_batch_norm=True, residual_bool=False, temper_value=0.5
    ):

        super(EncoderLayer, self).__init__()

        # check what implementation of residual to use
        self.residual_bool = residual_bool

        # attention heads
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, scaled_dropout=scaled_dropout,
            use_batch_norm=use_batch_norm, residual_bool=residual_bool, temper_value=temper_value
        )

        # feed forward part
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None, position_dpa=None):  # enc_non_embedded, enc_input, enc_pos,

        verbose_sizes = False
        if position_dpa is not None and verbose_sizes:
            print("dpa in self_attn:", position_dpa.size())

        # here q, k, w are all the same at input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask, position_dpa=position_dpa
        )

        # do feed forward
        if self.residual_bool:  # use new residual implementation
            enc_output = self.pos_ffn(enc_output, enc_input)
        else:  # typical self-attention representation
            enc_output = self.pos_ffn(enc_output, None)

        return enc_output, enc_slf_attn

''' Define the Layers '''
import torch.nn as nn
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


# TODO: the original encoder has 2 layers! now we are using just one
class EncoderLayer(nn.Module):
    """
    Compose with two layers
    """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # attention heads
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )

        # feed forward part
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):

        # here q,k,w are all the same at input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask
        )

        # added w1 layer, trying to experiment with different outputs (not crucial here!)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

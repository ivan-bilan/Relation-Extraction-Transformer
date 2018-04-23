import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1, temper_value=0.5):
        super(ScaledDotProductAttention, self).__init__()

        # add temper as hyperparameter
        self.temper = np.power(d_model, temper_value)    # 0.5 originally
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None, position_dpa=None):

        # initial attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        verbose_sizes = False

        # work with diagonal positional encodings
        if position_dpa is not None:
            if verbose_sizes:
                print("using diagonal positional encodings 2")
                print()
                print("q.size()                    ", q.size())                             # [150, 86, 120]
                print("k.transpose(1, 2).size()    ", k.transpose(1, 2).size())             # [150, 120, 86]
                print("attn.size()                 ", attn.size())                          # [150, 86, 86]
                print("position_dpa.size()         ", position_dpa.size())                  # [150, 86, 120]
                print("position_dpa.transpose(1, 2)", position_dpa.transpose(1, 2).size())  # [150, 120, 86]
                print()

            # TODO: do we include temper here as well?
            attn_pos = torch.bmm(q, position_dpa.transpose(1, 2)) / self.temper

            # apply mask to the diagonal positional attention as well
            if attn_mask is not None:

                assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

                attn_pos.data.masked_fill_(attn_mask, -float('inf'))

            if verbose_sizes:
                print(attn_pos.size())   # [150, 86, 86]

        # print(attn)
        # print(type(attn), attn.size())

        # print(attn_mask)
        # attn_mask = None

        if attn_mask is not None:
            # print(attn_mask)
            # print(attn_mask.size(), attn.size())

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        # position attention shifted truncated
        if position_dpa is not None:

            def stripe(a):
                # get the diagonal matrix stripe
                a = np.ascontiguousarray(a)
                i, j = a.shape
                assert i >= j
                k, l = a.strides
                return np.lib.stride_tricks.as_strided(a, (i - j + 1, j), (k, k + l))

            # attn_pos = stripe(attn_pos)

            attn = torch.bmm(attn, attn_pos)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

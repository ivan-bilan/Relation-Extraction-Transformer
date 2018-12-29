import torch
import torch.nn as nn
import numpy as np

from global_random_seed import RANDOM_SEED
from utils.attention_investigation import investigate_attention

# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k, attn_dropout=0.1, temper_value=0.5):
        super().__init__()

        self.temper = np.power(d_k, temper_value)    # 0.5 originally
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None, position_dpa=None, sentence_words=None):

        # initial attention
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temper
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
            attn_pos = torch.bmm(q, position_dpa.transpose(1, 2))
            attn_pos = attn_pos / self.temper

            # apply mask to the diagonal positional attention as well
            if verbose_sizes:
                print(attn_pos.size())   # [150, 86, 86]

            def batch_stripe(a):
                """
                Get a diagonal stripe of a matrix m x n, where n > m
                this implementation also takes into account batched matrices,
                so the stripe is calculated over a batch x for a matrix of size[x, m, n]
                """
                # another solution
                # a = a[::-1]  # ValueError: negative step not yet supported
                # do the usual left top to right bottom
                # return a[::-1]

                b, i, j = a.size()
                assert i > j
                b_s, k, l = a.stride()

                # left top to right bottom
                return torch.as_strided(a, (b, i - j+1, j), (b_s, k, k + l))

                # left bottom to right top
                # a = a[..., j-1:, :]
                # return torch.as_strided(a, (b, i-j, j), (b_s, k, l-k))

            pre_processed = attn_pos.transpose(1, 2)
            do_flip = torch.flip(pre_processed, [2])

            # print(do_flip)
            attn_pos = batch_stripe(do_flip)

            investigate_attention_flag = False
            if investigate_attention_flag:
                investigate_attention(attn, attn_pos, sentence_words, self.vocab)

            if verbose_sizes:
                print(attn_pos.size())
                print(attn.size())
                print(attn_pos.transpose(1, 2).size())

            attn = attn + attn_pos.transpose(1, 2)

        if attn_mask is not None:
            attn = attn.masked_fill_(attn_mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn

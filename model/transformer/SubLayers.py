''' Define the sublayers in encoder/decoder layer '''

import numpy as np
import torch
import torch.nn as nn
from .Modules import ScaledDotProductAttention

from global_random_seed import RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, position_dpa=None, dropout=0.1, scaled_dropout=0.1,
                 use_batch_norm=True, residual_bool=False, temper_value=0.5
                 ):

        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.use_batch_norm = use_batch_norm
        self.residual_bool = residual_bool

        # TODO: default without cuda, do we need cuda call here?
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # TODO: try # , nonlinearity='relu'
        # nn.init.kaiming_normal_(self.w_qs.weight)  # xavier_normal used originally
        # nn.init.kaiming_normal_(self.w_ks.weight)  # xavier_normal
        # nn.init.kaiming_normal_(self.w_vs.weight)  # xavier_normal

        # new weight initialization as per:
        # https://github.com/jadore801120/attention-is-all-you-need-pytorch/commit/2077515a8ab24f4abdda9089c502fa14f32fc5d9
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # for relative positional encodings
        # self.pos_dpa = nn.Linear(d_model, n_head * d_k)
        # nn.init.kaiming_normal_(self.pos_dpa.weight)

        # self.position_dpa2 = nn.Parameter(torch.FloatTensor(n_head, (96 * 2) - 1, d_k).cuda())

        # for dpa, fill with ones
        # self.dpa_qs = nn.Parameter(torch.FloatTensor(n_head, d_model*2, d_k).cuda())
        # init.constant(self.dpa_qs, 1)

        # TODO: test this, initially dropout was always set to 0.1!
        # TODO: higher makes the model stable, but Recall is now much lower!
        self.attention = ScaledDotProductAttention(d_k, scaled_dropout, temper_value)

        if self.use_batch_norm:  # batch norm
            self.layer_norm = nn.BatchNorm1d(d_model)
            # self.layer_norm = nn.GroupNorm(d_model, 42)
        else:  # layer norm
            self.layer_norm = nn.LayerNorm(d_model)

        # TODO: try with , bias=False
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, position_dpa_vector=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        sz_b, len_q, d_model = q.size()
        sz_b, len_k, d_model = k.size()
        sz_b, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # treat the result as a (n_head * mb_size) size batch
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # do the same as above but for the relative positional embeddings
        if position_dpa_vector is not None:
            sz_b1, len_q1, d_model = position_dpa_vector.size()
            position_dpa_vector = position_dpa_vector.view(sz_b1, len_q1, n_head, d_k)
            position_dpa_vector = position_dpa_vector.permute(2, 0, 1, 3).contiguous().view(-1, len_q1, d_k)

            # position_dpa = position_dpa.repeat(n_head, 1, 1)
            # position_dpa = position_dpa.view(n_head, -1, d_model//n_head)
            # position_dpa = position_dpa.view(-1, len_q * 2, d_k)

        if attn_mask is not None:
            if position_dpa_vector is not None:
                attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
                output, attns = self.attention(q, k, v,
                    attn_mask=attn_mask,
                    position_dpa=position_dpa_vector
                )
            else:
                attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
                output, attns = self.attention(q, k, v, attn_mask=attn_mask)

        # don't use masking if none given
        else:
            outputs, attns = self.attention(q, k, v)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        # project back to residual size
        # TODO: some people suggest to use bias=False when projecting!
        outputs = self.dropout(self.fc(output))

        if self.use_batch_norm:  # use batch norm
            # batch_norm expects (batch_size, h_units, seq_len), we have (batch_s, seq_len, h_units)
            outputs = outputs.permute(0, 2, 1)

            # have to make everything contiguous to make it run on CUDA
            if self.residual_bool:  # if new residual, add it only in PFF later
                outputs = self.layer_norm(outputs.contiguous())
            else:  # use typical self-attention implementation
                # TODO: make sure this actually works as it should
                outputs = self.layer_norm(outputs.contiguous() + residual.permute(0, 2, 1).contiguous())

            # move columns back
            return outputs.permute(0, 2, 1), attns

        else:  # use layer norm
            if self.residual_bool:  # if new residual, add it only in PFF later
                return self.layer_norm(outputs), attns
            else:
                return self.layer_norm(outputs + residual), attns


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise

        if self.use_batch_norm:
            self.layer_norm = nn.BatchNorm1d(d_hid)  # BatchNorm1d(d_hid)

            # other options here
            # self.layer_norm = nn.GroupNorm(d_hid, d_hid)
        else:
            self.layer_norm = nn.LayerNorm(d_hid)

        self.dropout = nn.Dropout(dropout)

        # instead of relu also tried: ELU, LeakyReLU, PReLU, ReLU6, RReLU, SELU
        self.relu = nn.RReLU()  # nn.ReLU() used originally

    def forward(self, x, residual=None):

        # redirect the residual from the MultiHeadAttention directly to the end of FFN if given one
        if residual is None:
            residual = x

        output = x.transpose(1, 2)
        output = self.w_2(self.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)

        if self.use_batch_norm:
            # batch_norm expects (batch_size, h_units, seq_len), we have (batch_s, seq_len, h_units)
            outputs = output.permute(0, 2, 1)
            residual_permuted = residual.permute(0, 2, 1)

            # have to make everything contiguous to make it run on CUDA
            outputs = self.layer_norm(outputs.contiguous() + residual_permuted.contiguous())
            # move columns back
            return outputs.permute(0, 2, 1)
        else:
            return self.layer_norm(output + residual)

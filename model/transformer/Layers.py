''' Define the Layers '''
import torch
import numpy as np
import torch.nn as nn
from .Constants import *
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)]
    )

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):

    ''' Indicate the padding-related part to mask '''

    # the input must be a non-embedded version of the sequence!

    """
    in the original implementation this is the input:
    # see full example under verbose_masking_input_example.txt
    
    batch size X sequence size
    
     best -proj_share_weightdata/multi30k.atok.low.pt -save_model trained -save_mode
    Namespace(batch_size=64, cuda=True, d_inner_hid=1024, d_k=64, d_model=512, d_v=64, d_word_vec=512, data='data/multi30k.atok.low.pt', dropout=0.1, embs_share_weight=False, epoch=10, log=None, max_token_seq_len=52, n_head=8, n_layers=6, n_warmup_steps=4000, no_cuda=False, proj_share_weight=True, save_mode='best', save_model='trained', src_vocab_size=2909, tgt_vocab_size=3149)
    [ Epoch 0 ]
      - (Training)   :   0%|          | 0/454 [00:00<?, ?it/s]src_seq Variable containing:
        2  2315   219  ...      0     0     0
        2  2315  2163  ...      0     0     0
        2  2315  1248  ...      0     0     0
           ...          ⋱          ...
        2  2315  2602  ...      0     0     0
        2  2315  2386  ...      0     0     0
        2  1047  1323  ...      0     0     0
    [torch.cuda.LongTensor of size 64x28 (GPU 0)]

    src_seq_type <class 'torch.autograd.variable.Variable'> torch.Size([64, 28])
    J:\Anaconda_Python3_6\envs\cuda\lib\site-packages\torch\nn\modules\module.py:325: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
      result = self.forward(*input, **kwargs)
    src_seq Variable containing:
        2  2315   433  ...      0     0     0
        2  2315   747  ...      0     0     0
        2  1694   358  ...      0     0     0
           ...          ⋱          ...
        2  2315  2297  ...      0     0     0
        2  1694  1047  ...      0     0     0
        2  1694  2537  ...      0     0     0
    [torch.cuda.LongTensor of size 64x25 (GPU 0)]
    
    src_seq_type <class 'torch.autograd.variable.Variable'> torch.Size([64, 25])
    src_seq Variable containing:
        2  2315   978  ...      0     0     0
        2  2566  1253  ...      0     0     0
        2  1253  1401  ...      0     0     0
           ...          ⋱          ...
        2  2315  2778  ...      0     0     0
        2   747   898  ...      0     0     0
        2  2315  2163  ...      0     0     0
    [torch.cuda.LongTensor of size 64x35 (GPU 0)]
    
    src_seq_type <class 'torch.autograd.variable.Variable'> torch.Size([64, 35])
      - (Training)   :   1%|          | 3/454 [00:02<05:11,  1.45it/s]src_seq Variable containing:
        2  2315  1492  ...      0     0     0
        2  2315   789  ...      0     0     0
        2  2315  1492  ...      0     0     0
           ...          ⋱          ...
        2  2315  1078  ...      0     0     0
        2  2315  2263  ...      0     0     0
        2  2315  2695  ...      0     0     0
    [torch.cuda.LongTensor of size 64x37 (GPU 0)]
    
    src_seq_type <class 'torch.autograd.variable.Variable'> torch.Size([64, 37])

    
    # output after masking
    (61,.,.) =
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
         ...       ⋱       ...
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
    
    (62,.,.) =
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
         ...       ⋱       ...
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
       0   0   0  ...    1   1   1
    
    (63,.,.) =
       0   0   0  ...    0   0   1
       0   0   0  ...    0   0   1
       0   0   0  ...    0   0   1
         ...       ⋱       ...
       0   0   0  ...    0   0   1
       0   0   0  ...    0   0   1
       0   0   0  ...    0   0   1
    [torch.cuda.ByteTensor of size 64x27x27 (GPU 0)]

    """
    """
    # original func
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)    # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask
    """

    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)    # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


# TODO: the original encoder has 2 layers! now we are using just one
class EncoderLayer(nn.Module):
    """
    Compose with two layers
    """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # the longest sentence
        n_position = 96 + 1
        # self.n_max_seq = 96

        # don't really need embeddings here,
        # this was done so that they can also be trained if needed?
        # not sure, follow this discussion: https://github.com/jadore801120/attention-is-all-you-need-pytorch/issues/33
        self.position_enc = nn.Embedding(n_position, d_model, padding_idx=PAD)  # here should be d_model = vec_size
        self.position_enc.weight.data = position_encoding_init(n_position, d_model)

        # attention heads
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )

        # feed forward part
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout
        )

    def forward(self, enc_non_embedded, enc_input, enc_pos, slf_attn_mask=None):
        """
        :param enc_input:
        :param enc_pos:
        :param slf_attn_mask:
        :return:
        """

        # TODO: try adding vectors (word vec + pos vec) instead of just appending them
        # TODO: also try with relative positions instead of absolute
        # TODO: try experimenting with character-based embeddings???
        # add positional encoding to the initial input, add emd_vec+pos_vec value by value
        enc_input += self.position_enc(enc_pos)

        # TODO: this is implemented, but P=1 R=0 as a result of using it
        # print(enc_input.size(), type(enc_input))
        # print(enc_input)
        if slf_attn_mask:
            slf_attn_mask = get_attn_padding_mask(enc_non_embedded, enc_non_embedded)

        # here q, k, w are all the same at input
        # TODO: masking not implemented
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

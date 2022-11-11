from global_random_seed import RANDOM_SEED

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import constant, torch_utils
from .transformer.Models import Encoder

# make everything reproducible
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class RelationModel(object):
    """
    A wrapper class for the training and evaluation of models.
    """

    def __init__(self, opt, emb_matrix=None):

        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)

        # pass weights per class, each class corresponds to its index
        weights = [opt['weight_no_rel']]
        rel_classes_weights = [opt["weight_rest"]] * 41
        weights.extend(rel_classes_weights)

        print("Using weights", weights)
        assert len(weights) == 42

        class_weights = torch.FloatTensor(weights).to("cuda")

        self.criterion = nn.CrossEntropyLoss(class_weights)  # weight=class_weights
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # print(self.parameters)
        # print(len(self.parameters))

        if opt['cuda']:
            self.model.to("cuda")
            self.criterion.to("cuda")

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        """
        Run a step of forward and backward model update.
        """

        if self.opt['cuda']:
            inputs = [b.to("cuda") for b in batch[:9]]
            labels = batch[9].to("cuda")
        else:
            inputs = [b for b in batch[:9]]
            labels = batch[9]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        # print(labels)
        loss = self.criterion(logits, labels)

        # backward step
        loss.backward()

        # do gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()

        loss_val = loss.item()

        return loss_val

    def predict(self, batch, unsort=True):
        """
        Run forward prediction. If unsort is True, recover the original order of the batch.
        """

        if self.opt['cuda']:
            inputs = [b.to("cuda") for b in batch[:9]]
            labels = batch[9].to("cuda")
        else:
            inputs = [b for b in batch[:9]]
            labels = batch[9]

        orig_idx = batch[10]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)

        probs = F.softmax(logits, dim=-1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]

        return predictions, probs, loss.item()

    def update_lr(self, new_lr):
        """
        Update learning rate of the optimizer
        :param new_lr: new learning rate
        """
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        """
        Save the model to a file
        :param filename:
        :param epoch:
        :return:
        """
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size, opt):
        super(PositionAwareAttention, self).__init__()

        self.opt = opt
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=True)

        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=True)
        else:
            self.wlinear = None

        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):

        # TODO: experiment with he and xavier
        # done, not really helping in any way here
        self.ulinear.weight.data.normal_(std=0.001).to("cuda")
        self.vlinear.weight.data.normal_(std=0.001).to("cuda")
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001).to("cuda")

        self.tlinear.weight.data.zero_().to("cuda")  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f, lstm_units=None, lstm_layer=False):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """

        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)

        # TODO: vlinear vs ulinear, u works better, but does it make sense to share those weights?
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
            batch_size, seq_len, self.attn_size
        )

        """
        # q_proj done step by step here to catch errors better
        # info on view and unsqueeze - 
        # https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155

        # q.size = 50x200
        q_proj = self.vlinear(q.view(-1, self.query_size))  # 50x200
        q_proj = q_proj.contiguous()  # 50x200
        q_proj = q_proj.view(batch_size, self.attn_size)  # <-- this is were size error happens  # 50x200
        q_proj = q_proj.unsqueeze(1)  # 50x200, adds new dimension
        q_proj = q_proj.expand(batch_size, seq_len, self.attn_size)  # 50x91x200 batch x seq_size x hidden size
        """

        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size
            )
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]

        # view in PyTorch is like reshape in numpy, view(n_rows, n_columns)
        # view(-1, n_columns) - here we define the number of columns, but n_rows will be chosen by PyTorch
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(batch_size, seq_len)

        # mask padding
        # print(x_mask, x_mask.size())
        # print(x_mask.data)

        # fill elements of self tensor with value where mask is one
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=-1)

        # weighted average input vectors

        # to calculate final sentence representation z,
        # we first test two variants:

        # 1. use self-attention to calculate a_i and use lstm to get h_i
        if lstm_layer:
            outputs = weights.unsqueeze(1).bmm(lstm_units).squeeze(1)
        # 2. use self-attention for a_i and also for h_i
        else:
            outputs = weights.unsqueeze(1).bmm(x).squeeze(1)

        return outputs


class PositionAwareRNN(nn.Module):
    """
    A sequence model for relation extraction.
    """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()

        self.drop = nn.Dropout(opt['dropout'])
        self.drop_rnn = nn.Dropout(opt['lstm_dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)

        # part of speech embeddings
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(
                len(constant.POS_TO_ID), opt['pos_dim'],
                padding_idx=constant.PAD_ID
            )

        # NER embeddings
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID)

        # add all embedding sizes to have the final input size
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        print("Number of heads: ", opt["n_head"])
        print("d_v and d_k: ", input_size / opt["n_head"])

        # make sure the head units add up to n_model in integers
        assert (int(input_size / opt["n_head"])) * opt["n_head"] == input_size

        self.self_attention_encoder = Encoder(
            n_src_vocab=55950,  # vocab size
            n_max_seq=96,  # max sequence length in the dataset
            n_layers=opt["num_layers_encoder"],  # multiple attention+ffn layers
            d_word_vec=input_size,  # TODO: increase dim by pos and ner
            d_model=input_size,  # d_model has to equal embedding size
            d_inner_hid=opt["hidden_self"],  # original paper: n_model * 2
            n_head=opt["n_head"],  # number of heads                    # 8
            d_k=int(input_size / opt["n_head"]),  # this should be d_model / n_heads   # 40
            d_v=int(input_size / opt["n_head"]),  # this should be d_model / n_heads   # 40
            dropout=opt['dropout'],
            scaled_dropout=opt['scaled_dropout'],
            obj_sub_pos=opt['obj_sub_pos'],  # list of obj/subj positional encodings
            use_batch_norm=opt['use_batch_norm'],
            residual_bool=opt["new_residual"],
            diagonal_positional_attention=opt["diagonal_positional_attention"],
            relative_positions=opt["relative_positions"],
            temper_value=opt["temper_value"]
        )

        # initial implementation with LSTM
        self.rnn = nn.LSTM(
            input_size,
            opt['hidden_dim'],
            opt['num_layers'],  # original 2
            batch_first=True,
            dropout=opt['lstm_dropout']
        )

        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['attn']:
            self.attn_layer = PositionAwareAttention(
                input_size=input_size,  # 360, hidden_dim originally, but for self-attention should be inout_size?
                query_size=opt["query_size_attn"],  # 360  # doesnt have to equal input size  # 200 for lstm
                feature_size=2 * opt['pe_dim'],
                attn_size=opt['attn_dim'],  # 360, attn_dim    # doesnt have to equal input size
                opt=opt
            )

            # TODO: does this work with relative positions?
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.linear.weight, gain=1)  # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # decide fine-tuning
        if self.topn <= 0:
            print("Do not fine-tune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Fine-tune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Fine-tune all embeddings.")

    def zero_state(self, batch_size):
        """
        Initialize zero states for LSTM's hidden layer and cell
        """
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.to("cuda"), c0.to("cuda")
        else:
            return h0, c0

    def forward(self, inputs):

        words, masks, pos, ner, deprel, subj_pos, obj_pos, modified_pos_vec, inst_position = inputs  # unpack

        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())

        # data is split into batches in the loader.py
        batch_size = words.size()[0]

        # embedding lookup
        word_inputs = self.emb(words)

        # TODO: move data to cuda? does it affect performance at all?
        # word_inputs = word_inputs.cuda()

        # add part-of-speech and NER embeddings to the input matrix
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]

        # input_size = inputs.size(2)
        # input_size = len(words)

        # TODO: dpa fix -1, 0, 1
        # TODO: safest number to add? maxlen doesn't work here at all
        # modified_pos_vec = modified_pos_vec + 60
        # print(modified_pos_vec)

        if self.opt["self_att"] is True:

            inputs_self = self.drop(torch.cat(inputs, dim=2))  # add dropout to input # cat - concatenates seq
            inputs_self = inputs_self.to("cuda")
            words = words.to("cuda")

            if self.opt["relative_positions"]:
                subj_pos_inputs = subj_pos + 10  # constant.MAX_LEN
                obj_pos_inputs = obj_pos + 10  # constant.MAX_LEN  # +42  # 10 is best! # 5 doesnt work anymore
            else:
                subj_pos_inputs = subj_pos + constant.MAX_LEN
                obj_pos_inputs = obj_pos + constant.MAX_LEN

            # should add forward here based on:
            # https://github.com/huajianjiu/attention-is-all-you-need-pytorch/commit/ed2057a741f416b5a3843fded91f84eb72414944
            # TODO: not sure why? is this needed
            outputs, enc_slf_attn = self.self_attention_encoder.forward(
                enc_non_embedded=words,
                src_seq=inputs_self, src_pos=inst_position,
                pe_features=[subj_pos_inputs, obj_pos_inputs, modified_pos_vec]
            )

            # hidden should be of size --> batch_size x hidden_size (e.i. 50x200)
            outputs_h = outputs.permute(0, 2, 1)
            hidden = F.max_pool1d(outputs_h, kernel_size=outputs_h.size()[-1]).squeeze()
            # outputs of the self-attention network
            outputs = self.drop(outputs)

            # use self-attention for outputs and LSTM for the last hidden layer
            if self.opt["self_att_and_rnn"] is True:
                # get the LSTM layer
                h0, c0 = self.zero_state(batch_size)

                inputs_rnn_drop = self.drop_rnn(torch.cat(inputs, dim=2))
                inputs_rnn = nn.utils.rnn.pack_padded_sequence(inputs_rnn_drop, seq_lens, batch_first=True)
                outputs_rnn, (ht, ct) = self.rnn(inputs_rnn, (h0, c0))
                # retrieve original sequence after doing pack_padded_sequence
                outputs_rnn, _ = nn.utils.rnn.pad_packed_sequence(outputs_rnn, batch_first=True)
                hidden_rnn = self.drop_rnn(ht[-1, :, :])  # get the outmost layer h_n

                # ignore outputs
                outputs_rnn = self.drop_rnn(outputs_rnn)

        else:
            # use LSTM instead of self-attention

            inputs_rnn_drop = self.drop_rnn(torch.cat(inputs, dim=2))

            # for original LSTM
            # inputs: torch.Size([50, 74, 360])  # 2nd element is variable
            # hidden: torch.Size([50, 200]) # batch size, hidden size
            # outputs: torch.Size([50, 74, 200])

            # use rnn
            h0, c0 = self.zero_state(batch_size)
            # instead of padding, you have to pack and pad in pytorch
            # more at https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099/2
            inputs = nn.utils.rnn.pack_padded_sequence(inputs_rnn_drop, seq_lens, batch_first=True)
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
            # retrieve original sequence after doing pack_padded_sequence
            outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            hidden = self.drop(ht[-1, :, :])  # get the outmost layer h_n
            outputs = self.drop(outputs)

        # attention
        if self.opt['attn']:
            # converts all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101

            # when using relative positional encodings, maybe it's better to just add +10?
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)  # + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)  # + constant.MAX_LEN)

            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)

            if self.opt["self_att_and_rnn"] is True:
                final_hidden = self.attn_layer(outputs, masks, hidden, pe_features, outputs_rnn, lstm_layer=True)
            else:
                final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)

        else:
            # skip positional attention and only use the self-attention's maxpool representation
            final_hidden = hidden

        logits = self.linear(final_hidden)

        return logits, final_hidden

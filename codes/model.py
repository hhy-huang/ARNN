# coding: utf-8
from __future__ import print_function
from __future__ import division

import os

import numpy
from sklearn import neighbors
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size, loc_emd_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.loc_emb_size = loc_emd_size
        self.weight = nn.Parameter(torch.FloatTensor(self.loc_emb_size, self.loc_emb_size))
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))
        torch.nn.init.xavier_uniform(self.weight)

    def forward(self, loc_emb, loc_neigh_emb):
        # (batch, num, neigh_num, hidden_size) => (batch, num, hidden_size, neigh_num)
        loc_neigh_emb_t = loc_neigh_emb.permute(0, 1, 3, 2)
        # (batch, num, hidden_size) => (batch, num, 1, hidden_size)
        loc_emb = torch.unsqueeze(loc_emb, 2)
        # (batch, num, 1, neigh_num)
        weight = torch.matmul(loc_emb, loc_neigh_emb_t)
        weight = F.softmax(weight, dim=3)
        # (batch, num, 1, hidden_size)
        result = torch.matmul(weight, loc_neigh_emb)
        # (batch, num, hidden_size)
        result = torch.squeeze(result, 2)
        return result


class NeighborsAttn(nn.Module):
    """
    rnn model with neighbors attention
    """
    def __init__(self, parameters):
        super(NeighborsAttn, self).__init__()
        self.batch_size = parameters.batch_size
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.word_size = parameters.word_size
        self.word_emb_size = parameters.voc_emb_size
        self.user_size = parameters.uid_size
        self.user_emb_size = parameters.uid_emb_size
        # 隐藏层大小（默认）设置为100
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type
        self.neighbors_num = 10

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_word = nn.Embedding(self.word_size, self.word_emb_size)
        self.emb_user = nn.Embedding(self.user_size, self.user_emb_size)

        self.attn = Attn(self.attn_type, self.hidden_size, self.loc_emb_size)
        input_size = 2 * self.loc_emb_size + self.tim_emb_size + self.word_emb_size
        # num_layers = 1 （层数为1）
        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)

        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.O1 = nn.Linear(self.hidden_size, self.user_emb_size)
        self.O2 = nn.Linear(2 * self.user_emb_size, self.loc_size)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, u_list, loc, tim, word, squence_length, neighbors_dict):
        # 初始化循环神经网络中每个单元的两个隐层输出（hidden_output和cell_hidden_output）
        # num_layers * num_directions, batch, hidden_size
        h1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        h2 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c2 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        # 对地点、时间、语义、用户进行嵌入，并将前三者合并为一个
        user = torch.tensor(u_list, dtype=torch.long).cuda()
        loc_emb = self.emb_loc(loc).squeeze()  # [batch_size, seq_len, 1] => [batch_size, seq_len, loc_embedding_size]
        tim_emb = self.emb_tim(tim).squeeze()  # [batch_size, seq_len, 1] => [batch_size, seq_len, tim_embedding_size]
        word_emb = self.emb_word(word).squeeze()
        user_emb = self.emb_user(user).squeeze()
        # 这里的x是最终要被concatenate然后输入LSTM的
        x = torch.cat((loc_emb, tim_emb, word_emb),
                      2)  # [seq_len, 1, tim_embedding_size + loc_embedding_size + word_embedding_size]

        # 过滤一下neighbors_dict里面的元素，数据处理的遗留问题，将neighbors的数量一致为self.neighbors_num
        for key, llist in zip(neighbors_dict.keys(), neighbors_dict.values()):
            llist_copy = llist.copy()
            for item in llist_copy:
                if item >= self.loc_size:
                    neighbors_dict[key].remove(item)
            top_k_list = Counter(neighbors_dict[key]).most_common(self.neighbors_num)  # 出现次数最多的前n个neighbors
            new_list = []
            for tup in top_k_list:
                new_list.append(tup[0])
            neighbors_dict[key] = new_list

        # 对neighbors进行attention
        neighbors_tensor = {}
        for pid in neighbors_dict:
            tmp = torch.LongTensor(neighbors_dict[pid]).cuda().reshape(len(neighbors_dict[pid]), 1)
            neighbors_tensor[pid] = []
            for i in range(len(tmp)):
                neighbor_emb = self.emb_loc(tmp[i]).squeeze().tolist()
                neighbors_tensor[pid].append(neighbor_emb)

        # create loc_neigh_emb
        loc_neigh_emb = numpy.float32(
            numpy.zeros((self.batch_size, loc_emb.size()[1], self.neighbors_num, self.loc_emb_size)))
        for bt in range(loc_emb.size()[0]):
            for poi in range(loc_emb.size()[1]):
                pid = int(loc[bt][poi])
                if pid == 0:  # pad
                    neighbor_emb = [[0] * self.loc_emb_size] * self.neighbors_num
                else:
                    neighbor_emb = neighbors_tensor[pid]
                loc_neigh_emb[bt][poi] = neighbor_emb
        loc_neigh_emb = torch.tensor(loc_neigh_emb).cuda()
        ck = self.attn(loc_emb, loc_neigh_emb)  # (batch_size, num, hidden_size)
        # 拼接为(elk, etk, esk, ck)
        x = torch.cat((x, ck), 2)
        x = self.dropout(x)
        length = len(nn.utils.rnn.pack_padded_sequence(x, squence_length, batch_first=True, enforce_sorted=False)[1])
        x = nn.utils.rnn.pack_padded_sequence(x, squence_length, batch_first=True,
                                              enforce_sorted=False)  # 这里压紧是为了输入LSTM

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_state, h1 = self.rnn_encoder(x, h1)
        elif self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_encoder(x, (h2, c2))  # hidden_state是所有隐藏状态的tensor，所以这里取hk

        hk = nn.utils.rnn.pad_packed_sequence(hidden_state, batch_first=True)[0]  # (128,10,100)
        O_K = self.O1(hk.reshape(self.batch_size, length, self.hidden_size))
        # broadcasting一下，补全sequence的长度(batch_size, sequence_len, user_emb_size)
        user_emb = user_emb.reshape(self.batch_size, 1, self.user_emb_size) + torch.zeros(self.batch_size, length,
                                                                                          self.user_emb_size).cuda()
        tmp = torch.cat((O_K, user_emb), 2)  # (128,10,80)
        O_k2 = self.O2(tmp)
        Yk = F.log_softmax(O_k2)

        return nn.utils.rnn.pack_padded_sequence(
            Yk, squence_length, batch_first=True, enforce_sorted=False
        ).data, Yk

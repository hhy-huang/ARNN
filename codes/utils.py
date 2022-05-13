import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader


class PaddingData:
    def __init__(self, data):
        self.data = data

    def get_padding_data(self, run_queue, batch_size):
        """
        获取批次数据并进行补全
        :return: 一个可以返回一个batch的DataLoader,每次返回经过补全的原序列id, 标签以及原序列的长度
        """
        data = self.data

        def collate_fn(run_queue):
            """
            根据输入的训练数据中最长的序列,padding其他序列
            :param run_queue 字典，包含`data`和`run_queue`
            :return u_list, target_batch_ori, loc_batch, tim_batch, target_batch, sequence_length
            """
            run_queue = run_queue
            u_list = [u for u, _ in run_queue]

            # loc_batch = pad_sequence([data[u][i]['loc'][:-data[u][i]['target'].data.size()[0]] for u, i in run_queue],
            #                          batch_first=True, padding_value=0)
            loc_batch = pad_sequence([data[u][i]['loc'] for u, i in run_queue], batch_first=True, padding_value=0)
            # tim_batch = pad_sequence([data[u][i]['tim'][:-data[u][i]['target'].data.size()[0]] for u, i in run_queue],
            #                          batch_first=True, padding_value=0)
            tim_batch = pad_sequence([data[u][i]['tim'] for u, i in run_queue], batch_first=True, padding_value=0)
            # word_batch = pad_sequence([data[u][i]['word'][:-data[u][i]['target'].data.size()[0]] for u, i in run_queue],
            #                          batch_first=True, padding_value=0)
            word_batch = pad_sequence([data[u][i]['word'] for u, i in run_queue], batch_first=True, padding_value=0)
            target_batch = pad_sequence([data[u][i]['target'] for u, i in run_queue], batch_first=True, padding_value=0)
            sequence_length = [data[u][i]['target'].data.size()[0] for u, i in run_queue]
            sequence_length_train = [data[u][i]['loc'].data.size()[0] - data[u][i]['target'].data.size()[0] for u, i in run_queue]
            f = torch.LongTensor
            return \
                u_list, \
                target_batch, \
                f(loc_batch).cuda(), \
                f(tim_batch).cuda(), \
                f(word_batch).cuda(), \
                f(pack_padded_sequence(
                    target_batch, sequence_length, batch_first=True, enforce_sorted=False
                ).data).cuda(), \
                f(sequence_length)

        return DataLoader(run_queue, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

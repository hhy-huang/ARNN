# coding: utf-8
from __future__ import division, print_function

import argparse
import json
import os
import time
from json import encoder

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import warnings

from model import NeighborsAttn
from train import (RnnParameterData, generate_neighbors_dict,
                   generate_input_long_history,
                   generate_input_history,
                   markov, run_simple)

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
def run(args):
    parameters = RnnParameterData(batch_size=args.batch_size, loc_emb_size=args.loc_emb_size,
                                  uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
    argv = {'batch_size': args.batch_size,
            'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    print('*' * 15 + 'start training' + '*' * 15)
    print('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    # 选的是这个
    model = NeighborsAttn(parameters=parameters).cuda()
    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'

    # 标准（损失函数）
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    # 学习率自动调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-4)

    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [],
               'accuracy_1': [], 'accuracy_5': [], 'accuracy_10': [],
               'ndcg_1': [], 'ndcg_5': [], 'ndcg_10': [],
               'valid_acc1': {}, 'valid_acc5': {}, 'valid_acc10': {},
               'valid_ndcg1': {}, 'valid_ndcg5': {}, 'valid_ndcg10': {}}

    # 读取用户id列表
    candidate = parameters.data_neural.keys()
    # 将数据输入到传统马尔可夫模型中，获取其传统性能指标
    avg_acc_markov, users_acc_markov = markov(parameters, candidate)
    metrics['markov_acc'] = users_acc_markov

    data_train, train_idx = generate_input_history(parameters.data_neural, 'train', argv['history_mode'], candidate=candidate)
    data_test, test_idx = generate_input_history(parameters.data_neural, 'test', argv['history_mode'], candidate=candidate)

    print('data_name:{}'.format(args.data_name))
    print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    # if not os.path.exists(SAVE_PATH + tmp_path):
    #     os.mkdir(SAVE_PATH + tmp_path)

    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy_1': [], 'ndcg_1': [],
                    'accuracy_5': [], 'ndcg_5': [], 'accuracy_10': [], 'ndcg_10': []}
    neighbors_dict = generate_neighbors_dict(parameters.vid_list, parameters.path_data)
    # neighbors_dict = {}5
    # model.load_state_dict(torch.load('../results/checkpoint/ep_2.m'))
    # 开始训练
    for epoch in range(parameters.epoch):
        st = time.time()
        # 若此时没有预训练模型则开始训练
        # print(model)
        if args.pretrain == 0:
            model, avg_loss = run_simple(epoch, neighbors_dict, data_train, train_idx, 'train', lr, parameters.clip,
                                         model, optimizer,
                                         criterion, parameters.model_mode, batch_size=args.batch_size, )
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)

        avg_loss, avg_acc, avg_acc5, avg_acc10, users_acc1, users_acc5, users_acc10, avg_ndcg, avg_ndcg5, avg_ndcg10, \
        users_ndcg1, users_ndcg5, users_ndcg10 = run_simple(
            epoch, neighbors_dict, data_test, test_idx, 'test', lr, parameters.clip, model,
            optimizer, criterion, parameters.model_mode,
            batch_size=args.batch_size)
        print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))

        # 保存评价指标以及当前Epoch下的模型
        metrics['valid_loss'].append(avg_loss)

        metrics['accuracy_1'].append(avg_acc)
        metrics['valid_acc1'][epoch] = users_acc1
        metrics['ndcg_1'].append(avg_ndcg)
        metrics['valid_ndcg1'][epoch] = users_ndcg1

        metrics['accuracy_5'].append(avg_acc5)
        metrics['valid_acc5'][epoch] = users_acc5
        metrics['ndcg_5'].append(avg_ndcg5)
        metrics['valid_ndcg5'][epoch] = users_ndcg5

        metrics['accuracy_10'].append(avg_acc10)
        metrics['valid_acc10'][epoch] = users_acc10
        metrics['ndcg_10'].append(avg_ndcg10)
        metrics['valid_ndcg10'][epoch] = users_ndcg10

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        epoch_str = '%d' % epoch
        json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + epoch_str + '.txt', 'w'), indent=4)

        # 更新学习率，并通过学习率判断是否回退模型（类似早停）
        scheduler.step(avg_acc)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy_1'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break
        if args.pretrain == 1:
            break

    # 保存最终模型
    mid = np.argmax(metrics['accuracy_1'])
    avg_acc = metrics['accuracy_1'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)

    for key in metrics_view:
        metrics_view[key] = metrics[key]
    print('argv:')
    print(argv)
    print('metrics_view:')
    print(metrics_view)
    # print('metrics:')
    # print(metrics)
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')

    # 移除过程文件
    # for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
    #     for name in files:
    #         remove_path = os.path.join(rt, name)
    #         os.remove(remove_path)
    # os.rmdir(SAVE_PATH + tmp_path)
    return avg_acc


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 0


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.set_device(2)
    # ptvsd.enable_attach(address = ('115.156.96.2', 3000))
    # ptvsd.wait_for_attach()
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=100, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare_NYC_4input')
    parser.add_argument('--learning_rate', type=float, default=1 * 1e-3)
    parser.add_argument('--lr_step', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--model_mode', type=str, default='attn_local_long')
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    ours_acc = run(args)

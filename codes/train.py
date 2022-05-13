# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from tqdm import tqdm
from torch.autograd import Variable

import numpy as np
import pickle
from collections import deque
from collections import Counter
from utils import PaddingData

neighbors_num = 60
batch_size = 64


class RnnParameterData(object):
    def __init__(self, batch_size=128, loc_emb_size=100, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10,
                 hidden_size=100,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=50, rnn_type='LSTM', model_mode="simple",
                 data_path='../data', save_path='../results', data_name='/foursquare_NYC_4input'):
        self.batch_size = batch_size
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name

        # 将目标文件反序列化。将文件中的数据解析为一个Python对象。
        # data = pickle.load(open(self.data_path + self.data_name + '.pkl', 'rb'))
        data = pickle.load(open('../data/foursquare_NYC_4input' + '.pkl', 'rb'))
        self.path_data = pickle.load(open('../data/paths_NYC' + '.pkl', 'rb'))

        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.wid_list = data['wid2word_list']
        self.data_neural = data['data_neural']

        self.tim_size = 48  # 时间间隔定为
        self.loc_size = len(self.vid_list)  #
        self.uid_size = len(self.uid_list)
        self.word_size = len(self.wid_list)

        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size

        self.hidden_size = hidden_size  # 隐藏层定为100

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode


def generate_neighbors_dict(vid_list, paths_data):
    neighbors_dict = {}
    LL_path = paths_data['LL']
    LUL_path = paths_data['LUL']
    LVL_path = paths_data['LVL']
    for vid in vid_list.values():
        if vid[0] == 0:
            continue
        vid = vid[0]
        neighbors_dict[vid] = []
    # LL
    for vid in vid_list.values():
        if vid[0] == 0:
            continue
        vid = vid[0]
        for path in LL_path[vid]:
            neighbors_dict[vid].extend(path)  # 加入neighbors
            # neighbors_dict[vid] = list(set(neighbors_dict[vid]))

    # LUL
    for vid in vid_list.values():
        if vid[0] == 0:
            continue
        vid = vid[0]
        for path in LUL_path[vid]:
            neighbors_dict[vid].extend(path)
            # neighbors_dict[vid] = list(set(neighbors_dict[vid]))

    # LVL
    for vid in vid_list.values():
        if vid[0] == 0:
            continue
        vid = vid[0]
        for path in LVL_path[vid]:
            neighbors_dict[vid].extend(path)
            # neighbors_dict[vid] = list(set(neighbors_dict[vid]))

    for vid in vid_list.values():
        if vid[0] == 0:
            continue
        vid = vid[0]
        top_k_list = Counter(neighbors_dict[vid]).most_common(neighbors_num)  # 出现次数最多的前n个neighbors
        new_list = []
        for tup in top_k_list:
            new_list.append(tup[0])
        neighbors_dict[vid] = new_list
    return neighbors_dict


# 一个用户一条长序列
def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions_with_word']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:  # 不要第一个
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1], s[2]) for s in sessions[tt]])
            for j in range(c):  # 0-当前的train id
                history.extend([(s[0], s[1], s[2]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            history_word = np.reshape(np.array([s[2] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_word'] = Variable(torch.LongTensor(history_word))
            trace['history_count'] = history_count

            loc_tim = history
            loc_tim.extend([(s[0], s[1], s[2]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            word_np = np.reshape(np.array([s[2] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['word'] = Variable(torch.LongTensor(word_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

# 一个用户多条序列
def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions_with_word']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            word_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['word'] = Variable(torch.LongTensor(word_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1], s[2]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1], s[2]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            history_word = np.reshape(np.array([s[2] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_word'] = Variable(torch.LongTensor(history_word))
            if mode2 == 'avg':
                trace['history_count'] = history_count
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    # train_idx：字典，键为user的id，值为该user的涉及到的轨迹点
    user = list(train_idx.keys())
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)  # 打乱user
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_ndcg(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    # acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return ndcg


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def run_simple(epoch, neighbors_dict, data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None,
               batch_size=128):
    """
        运行训练和测试等操作
        mode=train: return model, avg_loss
        mode=test: return avg_loss,avg_acc,users_rnn_acc
    """
    global acc
    run_queue = None
    # 获取数据队列
    if mode == 'train':
        model.train(True)
        # `random`说明数据被打乱
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []

    users_acc = {}
    users_ndcg = {}
    padding_data_iter = PaddingData(data)
    total_batch = len(padding_data_iter.get_padding_data(run_queue, batch_size))
    num = 1
    for u_list, target_batch_ori, loc_batch, tim_batch, word_batch, target_batch, sequence_length in tqdm(
            padding_data_iter.get_padding_data(
                    run_queue,
                    batch_size)):
        for u in u_list:
            users_acc[u] = [0, 0, 0, 0]
            users_ndcg[u] = [0, 0, 0, 0]
        assert mode2 == 'attn_local_long'
        # if len(u_list) != batch_size:  # 最后无法构成一个batch的部分放弃
        #     continue
        scores, score = model(u_list, loc_batch, tim_batch, word_batch, sequence_length, neighbors_dict)
        # ?
        # if scores.data.size()[0] > target.data.size()[0]:
        #     scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target_batch)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            for index, u in enumerate(u_list):
                users_ndcg[u][0] += len(target_batch_ori[index])
                users_acc[u][0] += len(target_batch_ori[index])
                acc = get_acc(target_batch_ori[index], score[index])
                ndcg = get_ndcg(target_batch_ori[index], score[index])

                # Top-1
                users_acc[u][1] += acc[2]
                users_ndcg[u][1] += ndcg[2]
                # Top-5
                users_acc[u][2] += acc[1]
                users_ndcg[u][2] += ndcg[1]
                # top-10
                users_acc[u][3] += acc[0]
                users_ndcg[u][3] += ndcg[0]

        print("---------------- loss:" + str(loss.data.cpu().numpy()) + " (epoch:" + str(epoch) + "  batch:" + str(num)
              + "/" + str(total_batch) + ")")
        num += 1
        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss

    elif mode == 'test':
        users_rnn_acc_top1 = {}
        users_rnn_acc_top5 = {}
        users_rnn_acc_top10 = {}
        users_rnn_ndcg_top1 = {}
        users_rnn_ndcg_top5 = {}
        users_rnn_ndcg_top10 = {}

        for u in users_acc:
            # Top-1
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc_top1[u] = tmp_acc.tolist()[0]
            # Top-5
            tmp_acc = users_acc[u][2] / users_acc[u][0]
            users_rnn_acc_top5[u] = tmp_acc.tolist()[0]
            # Top-10
            tmp_acc = users_acc[u][3] / users_acc[u][0]
            users_rnn_acc_top10[u] = tmp_acc.tolist()[0]

        avg_acc_top1 = np.mean([users_rnn_acc_top1[x] for x in users_rnn_acc_top1])
        avg_acc_top5 = np.mean([users_rnn_acc_top5[x] for x in users_rnn_acc_top5])
        avg_acc_top10 = np.mean([users_rnn_acc_top10[x] for x in users_rnn_acc_top10])

        print('avg_acc_Top-1:{}'.format(avg_acc_top1))
        print('avg_acc_Top-5:{}'.format(avg_acc_top5))
        print('avg_acc_Top-10:{}'.format(avg_acc_top10))

        for u in users_ndcg:
            # Top-1
            tmp_ndcg = users_ndcg[u][1] / users_ndcg[u][0]
            users_rnn_ndcg_top1[u] = tmp_ndcg.tolist()[0]
            # Top-5
            tmp_ndcg = users_ndcg[u][2] / users_ndcg[u][0]
            users_rnn_ndcg_top5[u] = tmp_ndcg.tolist()[0]
            # Top-10
            tmp_ndcg = users_ndcg[u][3] / users_ndcg[u][0]
            users_rnn_ndcg_top10[u] = tmp_ndcg.tolist()[0]

        avg_ndcg_top1 = np.mean([users_rnn_ndcg_top1[x] for x in users_rnn_ndcg_top1])
        avg_ndcg_top5 = np.mean([users_rnn_ndcg_top5[x] for x in users_rnn_ndcg_top5])
        avg_ndcg_top10 = np.mean([users_rnn_ndcg_top10[x] for x in users_rnn_ndcg_top10])

        print('avg_ndcg_Top-1:{}'.format(avg_ndcg_top1))
        print('avg_ndcg_Top-5:{}'.format(avg_ndcg_top5))
        print('avg_ndcg_Top-10:{}'.format(avg_ndcg_top10))

        return avg_loss, avg_acc_top1, avg_acc_top5, avg_acc_top10, \
               users_rnn_acc_top1, users_rnn_acc_top5, users_rnn_acc_top10, \
               avg_ndcg_top1, avg_ndcg_top5, avg_ndcg_top10, \
               users_rnn_ndcg_top1, users_rnn_ndcg_top5, users_rnn_ndcg_top10


def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])  # POI序列
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc

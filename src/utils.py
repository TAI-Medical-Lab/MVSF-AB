from __future__ import print_function, division
import numpy as np
import torch
import torch.utils.data
from datetime import datetime
import csv
from Bio import SeqIO
from torch.optim import Optimizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, lchain, hchain, antigen, label):
        self.lchain = lchain
        self.hchain = hchain
        self.antigen = antigen
        self.label = label

    def __len__(self):
        return len(self.lchain)

    def __getitem__(self, i):
        return self.lchain[i], self.hchain[i], self.antigen[i], self.label[i]

def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    x2 = [a[2] for a in args]
    y = [a[3] for a in args]
    return x0, x1, x2, torch.stack(y, 0)

def log(m, file=None, timestamped=True, print_also=False):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "  # 代码根据timestamped参数决定是否在日志信息前面添加时间戳
    log_string = f"{curr_time if timestamped else ''}{m}"  # m是输入参数，代表要打印的日志信息
    if file is None:
        print(log_string)
    else:
        # file参数不为None，则将日志信息写入文件，并根据print_also参数决定是否同时打印到标准输出中
        print(log_string, file=file)
        if print_also:
            print(log_string)
        file.flush()  # 如果写入文件后，需要调用file.flush()方法将缓冲区中的数据刷新到磁盘上


def NormalizeData(train_tensor):
    min_val = torch.min(train_tensor)
    max_val = torch.max(train_tensor)
    # print('min,max:',min_val, max_val)
    normalized_train_tensor = (train_tensor - min_val) / (max_val - min_val)
    return normalized_train_tensor

# r2 score
def r_squared(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()  # 将张量转换为numpy数组
    y_pred = y_pred.cpu().detach().numpy()
    mean_value = np.mean(y_true)
    tss = np.sum((y_true - mean_value)**2)
    rss = np.sum((y_true - y_pred)**2)
    r_squared = 1 - rss / tss

    return r_squared

def pearsonr(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    mean_yt = np.mean(y_true)
    mean_yp = np.mean(y_pred)

    yt_dev = y_true - mean_yt
    yp_dev = y_pred - mean_yp

    numerator = np.sum(yt_dev * yp_dev)
    denominator = np.sqrt(np.sum(yt_dev ** 2)) * np.sqrt(np.sum(yp_dev ** 2))

    correlation = numerator / denominator

    return correlation


# get embedding dictionary
def embed_dict(fastaPath, embeddingPath):
    names = []
    for record in SeqIO.parse(fastaPath, "fasta"):
        names.append(record.name)
    embeddingPath = open(embeddingPath)
    csv_reader_lines = csv.reader(embeddingPath)
    seq_features = []
    for one_line in csv_reader_lines:
        one_line = list(map(float, one_line))
        # 把list转成tensor
        variable_line = torch.tensor(one_line, dtype=torch.float32)
        seq_features.append(variable_line)
    dictionary = dict(zip(names, seq_features[1:]))
    return dictionary

# get aaindex1 feature dictionary
def seq_aaindex_dict(protein_set,fastaPath,max_len=256):

    df = pd.read_csv('datasets/aaindex_pca.csv', header=0)
    # 选取的相关性较高的特征
    # 对除了第一列的每一列的值进行归一化
    # df = df.iloc[:, 1:]
    # df = (df - df.min()) / (df.max() - df.min())

    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[1:]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    # print(df)
    # feature_selected = [10, 13, 53, 55, 56, 57, 67, 70, 73, 112,
    #                     114, 130, 145, 150, 152, 209, 210, 212, 213, 214,
    #                     215, 216, 219, 220, 239, 257, 258, 260, 261, 262,
    #                     263, 264, 265, 266, 267, 268, 269, 272, 273, 275,
    #                     276, 277, 278, 279, 280, 285, 286, 287, 288, 289,
    #                     290, 291, 292, 293, 313, 315, 316, 317, 319, 338,
    #                     339, 340, 341, 343, 345, 346, 347, 348, 349, 350,
    #                     355, 360, 364, 380, 384, 386, 388, 390, 391, 396,
    #                     423, 434, 445, 446, 447, 483, 489, 509, 512, 515,
    #                     517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 530]
    feature_selected = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    # print(df.columns[19])
    for i in range(len(feature_selected)):
        feature_selected[i] = feature_selected[i]+1
    column_names = df.columns[feature_selected]  # 103
    # print(column_names)
    groups = df.groupby('AA')
    results_dict = {}  # aa对应的531特征 字典
    for group_name, group in groups:
        num_rows = group.shape[0]
        values_list = []
        for i in range(num_rows):
            row = group.iloc[i]
            values_list.append(row[column_names].tolist())
        results_dict[group_name] = values_list
    seq_dict = {}
    for protein in protein_set:
        seq = id2seq(protein, fastaPath)
        seq_feature = []
        for aa in seq:
            if aa in results_dict.keys():
                seq_feature.append(torch.tensor(results_dict[aa][0]))
        if len(seq_feature) < max_len:
            for i in range(max_len - len(seq_feature)):
                seq_feature.append(torch.zeros(len(feature_selected)))
        else:
            seq_feature = seq_feature[:max_len]
        seq_dict[protein] = torch.stack(seq_feature)
    return seq_dict

# get sequence from id
def id2seq(aa_id,fastaPath):
    with open(fastaPath) as f:
    # with open("datasets/seq_msa.fasta") as f:
        sequence_lines = []
        sequence_found = False
        for line in f:
            if line.startswith(">"):
                if aa_id in line:
                    sequence_found = True
            else:
                if sequence_found:
                    sequence_lines.append(line.strip())
                    break
        if sequence_found:
            return "".join(sequence_lines)
        else:
            return "A"

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):  # rho调整
        defaults = dict(rho=rho, **kwargs)
        self.base_optimizer = base_optimizer(params, **kwargs)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            grad_norm = torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SAM does not support sparse gradients')
                p_data = p.data
                grad_norm = torch.norm(grad)
                if grad_norm != 0:
                    grad /= grad_norm
                    perturbation = torch.zeros_like(p_data).normal_(0, rho*grad_norm)
                    p_data.add_(perturbation)
                    self.base_optimizer.step(closure)
                    p_data.sub_(perturbation)
        return loss

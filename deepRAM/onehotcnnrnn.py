import random
import numpy as np
import torch
import csv
import math
import time
from sklearn import metrics
from sklearn.model_selection import KFold
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import argparse
import warnings
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 全局变量
bases = 'ACGT'  # DNA bases
basesRNA = 'ACGU'  # RNA bases
batch_size = 128
data_type = 'DNA'
nummotif = 16  # 卷积核数量
motiflen = 5  # 卷积核长度
conv_layers = 1  # 卷积层数
RNN_type = 'BiLSTM'
RNN_layers = 1
RNN_hidden_size = 128
hidden_size = 64
dropprob = 0.3
dilation = 1


def seq_to_onehot(sequence):
    """将序列转换为one-hot编码矩阵"""
    base = bases if data_type == 'DNA' else basesRNA
    seq_len = len(sequence)

    # 创建4行L列的矩阵
    S = np.zeros((4, seq_len))

    for i, char in enumerate(sequence):
        if char == 'N' or char not in base:
            # 模糊碱基处理
            S[:, i] = [0.25, 0.25, 0.25, 0.25]
        else:
            for j, b in enumerate(base):
                if char == b:
                    S[j, i] = 1.0
    return S.T  # 转置为(seq_len, 4)

def logsampler(a, b):
    """对数空间采样"""
    x = np.random.uniform(low=0, high=1)
    y = 10 ** ((math.log10(b) - math.log10(a)) * x + math.log10(a))
    return y

def sqrtsampler(a, b):
    """平方根空间采样"""
    x = np.random.uniform(low=0, high=1)
    y = (b - a) * math.sqrt(x) + a
    return y



class CNNRNNNetwork(nn.Module):
    def __init__(self, nummotif, motiflen, conv_layers, RNN_hidden_size, hidden_size,
                 dropprob, dilation, RNN_type, RNN_layers):
        super(CNNRNNNetwork, self).__init__()

        # CNN部分参数
        self.nummotif = nummotif
        self.motiflen = motiflen
        self.conv_layers = conv_layers
        self.dilation = dilation
        self.dropprob = dropprob

        # RNN部分参数
        self.RNN_type = RNN_type
        self.RNN_layers = RNN_layers
        self.RNN_hidden_size = RNN_hidden_size

        # 输入通道数 (4 for one-hot)
        input_channels = 4

        # ========== 序列1的CNN部分 ==========
        self.conv1_layers = nn.ModuleList()
        self.bn1_layers = nn.ModuleList()

        # 第一层卷积
        self.conv1_layers.append(nn.Conv1d(
            in_channels=input_channels,
            out_channels=nummotif,
            kernel_size=motiflen,
            dilation=dilation,
            padding='valid'
        ))
        self.bn1_layers.append(nn.BatchNorm1d(nummotif))

        # 添加更多卷积层
        current_channels = nummotif
        for i in range(1, conv_layers):
            out_channels = nummotif * (2 ** i)  # 每层通道数翻倍
            self.conv1_layers.append(nn.Conv1d(  # 修正模块名称
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=3,  # 后续层使用较小的卷积核
                dilation=dilation,
                padding='valid'
            ))
            self.bn1_layers.append(nn.BatchNorm1d(out_channels))
            current_channels = out_channels

        # ========== 序列2的CNN部分 ==========
        self.conv2_layers = nn.ModuleList()
        self.bn2_layers = nn.ModuleList()

        # 第一层卷积
        self.conv2_layers.append(nn.Conv1d(
            in_channels=input_channels,
            out_channels=nummotif,
            kernel_size=motiflen,
            dilation=dilation,
            padding='valid'
        ))
        self.bn2_layers.append(nn.BatchNorm1d(nummotif))

        # 添加更多卷积层
        current_channels = nummotif
        for i in range(1, conv_layers):
            out_channels = nummotif * (2 ** i)  # 每层通道数翻倍
            self.conv2_layers.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=3,  # 后续层使用较小的卷积核
                dilation=dilation,
                padding='valid'
            ))
            self.bn2_layers.append(nn.BatchNorm1d(out_channels))
            current_channels = out_channels

        # ========== RNN部分 ==========
        # 序列1的RNN
        if RNN_type == 'LSTM':
            self.rnn1 = nn.LSTM(
                input_size=current_channels,
                hidden极_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )
        elif RNN_type == 'GRU':
            self.rnn1 = nn.GRU(
                input_size=current_channels,
                hidden_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )
        else:  # 默认使用BiLSTM
            self.rnn1 = nn.LSTM(
                input_size=current_channels,
                hidden_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )

        # 序列2的RNN
        if RNN_type == 'LSTM':
            self.rnn2 = nn.LSTM(
                input_size=current_channels,
                hidden_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )
        elif RNN_type == 'GRU':
            self.rnn2 = nn.GRU(
                input_size=current_channels,
                hidden_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )
        else:  # 默认使用BiLSTM
            self.rnn2 = nn.LSTM(
                input_size=current_channels,
                hidden_size=RNN_hidden_size,
                num_layers=RNN_layers,
                bidirectional=True,
                batch_first=True
            )

        # ========== 全连接层 ==========
        # 双向RNN的输出大小为2 * RNN_hidden_size
        # 两个序列拼接后大小为4 * RNN_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(4 * RNN_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropprob),
            nn.Linear(hidden_size, 1)
        )

        # 池化层 - 使用更小的核和步长
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 自适应池化层用于处理短序列
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropprob)


    def forward(self, x1, x2):
        # 输入形状: (batch, seq_len, 4)
        # 转置为CNN需要的形状: (batch, 4, seq_len)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        # ===== 序列1的CNN处理 =====
        for i in range(self.conv_layers):
            x1 = self.conv1_layers[i](x1)
            x1 = self.bn1_layers[i](x1)
            x1 = F.relu(x1)

            # 检查序列长度是否足够进行池化
            if x1.size(2) < 2:  # 如果序列长度小于2
                x1 = self.adaptive_pool(x1)  # 使用自适应池化
            else:
                x1 = self.pool(x1)  # 否则使用常规池化

            x1 = self.dropout(x1)

        # 确保序列长度至少为1
        if x1.size(2) < 1:
            x1 = self.adaptive_pool(x1)

        # 转置为RNN需要的形状: (batch, seq_len, channels)
        x1 = x1.permute(0, 2, 1)

        # ===== 序列2的CNN处理 =====
        for i in range(self.conv_layers):
            x2 = self.conv2_layers[i](x2)
            x2 = self.bn2_layers[i](x2)
            x2 = F.relu(x2)

            # 检查序列长度是否足够进行池化
            if x2.size(2) < 2:  # 如果序列长度小于2
                x2 = self.adaptive_pool(x2)  # 使用自适应池化
            else:
                x2 = self.pool(x2)  # 否则使用常规池化

            x2 = self.dropout(x2)

        # 确保序列长度至少为1
        if x2.size(2) < 1:
            x2 = self.adaptive_pool(x2)

        # 转置为RNN需要的形状: (batch, seq_len, channels)
        x2 = x2.permute(0, 2, 1)

        # ===== 序列1的RNN处理 =====
        # 确保序列长度至少为1
        seq_len1 = max(1, x1.size(1))
        seq_len2 = max(1, x2.size(1))

        if isinstance(self.rnn1, nn.LSTM):
            rnn1_out, (h_n1, c_n1) = self.rnn1(x1)
        else:
            rnn1_out, h_n1 = self.rnn1(x1)

        # 获取最后一个时间步的输出（双向）
        if self.rnn1.bidirectional:
            # 取前向最后一个时间步和后向第一个时间步
            forward1 = rnn1_out[:, -1, :self.RNN_hidden_size]
            backward1 = rnn1_out[:, 0, self.RNN_hidden_size:]
            rnn1_out = torch.cat((forward1, backward1), dim=1)
        else:
            rnn1_out = rnn1_out[:, -1, :]

        rnn1_out = self.dropout(rnn1_out)

        # ===== 序列2的处理 =====
        if isinstance(self.rnn2, nn.LSTM):
            rnn2_out, (h_n2, c_n2) = self.rnn2(x2)
        else:
            rnn2_out, h_n2 = self.rnn2(x2)

        # 获取最后一个时间步的输出（双向）
        if self.rnn2.bidirectional:
            # 取前向最后一个时间步和后向第一个时间步
            forward2 = rnn2_out[:, -1, :self.RNN_hidden_size]
            backward2 = rnn2_out[:, 0, self.RNN_hidden_size:]
            rnn2_out = torch.cat((forward2, backward2), dim=1)
        else:
            rnn2_out = rnn2_out[:, -1, :]

        rnn2_out = self.dropout(rnn2_out)

        # ===== 合并两个序列的特征 =====
        combined = torch.cat((rnn1_out, rnn2_out), dim=1)

        # ===== 全连接层 =====
        output = self.fc(combined)
        return torch.sigmoid(output)


class InteractionDataset():
    def __init__(self, filename):
        self.file = filename

    def openFile(self):
        """读取CSV文件并准备数据集"""
        dataset = []
        sequences1 = []
        sequences2 = []

        with open(self.file, 'r') as data:
            reader = csv.reader(data, delimiter=',')
            header = next(reader)  # 跳过标题行

            for row in reader:
                if len(row) < 3:
                    continue
                seq1, seq2, label = row[0], row[1], int(row[2])
                sequences1.append(seq1)
                sequences2.append(seq2)

                # 转换为one-hot编码
                x1 = seq_to_onehot(seq1)
                x2 = seq_to_onehot(seq2)
                dataset.append([[x1, x2], [label]])

        # 添加数据统计
        num_positive = sum(item[1][0] for item in dataset)
        print(f"Loaded {len(dataset)} samples, positive: {num_positive}, negative: {len(dataset) - num_positive}")

        return dataset, sequences1, sequences2


def k_fold_split(dataset, k=5):
    """将数据集划分为k折用于交叉验证"""
    kf = KFold(n_splits=k, shuffle=True)
    folds = []
    for train_idx, val_idx in kf.split(dataset):
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        folds.append((train_data, val_data))
    return folds


class InteractionDatasetClass(Dataset):
    """处理one-hot编码数据的数据集类"""

    def __init__(self, xy=None):
        # 找到最大序列长度
        max_len = max(max(el[0][0].shape[0], el[0][1].shape[0]) for el in xy)

        # 初始化列表存储填充后的数据
        x1_list = []
        x2_list = []

        for el in xy:
            # 对x1进行填充
            x1 = el[0][0]
            pad_width = max_len - x1.shape[0]
            if pad_width > 0:
                x1 = np.pad(x1, ((0, pad_width), (0, 0)), 'constant')
            x1_list.append(x1)

            # 对x2进行填充
            x2 = el[0][1]  # 修正索引错误
            pad_width = max_len - x2.shape[0]
            if pad_width > 0:
                x2 = np.pad(x2, ((0, pad_width), (0, 0)), 'constant')
            x2_list.append(x2)

        # 转换为numpy数组
        self.x1_data = np.stack(x1_list)
        self.x2_data = np.stack(x2_list)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)

        # 转换为torch张量
        self.x1_data = torch.from_numpy(self.x1_data).float()
        self.x2_data = torch.from_numpy(self.x2_data).float()
        self.y_data = torch.from_numpy(self.y_data).float()
        self.len = len(self.x1_data)
        self.seq_len = max_len
        self.input_size = self.x1_data.shape[2]  # 通常是4

    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def Load_Data(train_file):
    """加载数据并准备五折交叉验证"""
    print(f"Loading data from {train_file}")
    interaction_data = InteractionDataset(train_file)
    dataset, _, _ = interaction_data.openFile()

    # 五折交叉验证划分
    print("Creating 5-fold cross validation splits")
    folds = k_fold_split(dataset, k=5)

    return folds


def Calibration(folds):
    """使用五折交叉验证进行超参数校准"""
    print('Starting automatic calibration with 5-fold cross validation')



    # 超参数搜索空间
    RNN_hidden_size_list = [64, 128, 256]
    dropoutList = [0.3, 0.5]
    hidden_size_list = [64, 128]
    optim_list = ['Adam', 'AdamW']
    nummotif_list = [8, 16, 32]
    motiflen_list = [3]
    conv_layers_list = [1]
    dilation_list = [1, 2]

    num_epochs = 30  # 训练轮数

    # 存储每折的最佳结果
    fold_results = []
    best_auc_overall = 0
    best_hparams = None

    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\n=== Processing fold {fold_idx + 1}/5 ===')

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data)
        valid_dataset = InteractionDatasetClass(valid_data)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 随机采样超参数
        RNN_hidden_size = random.choice(RNN_hidden_size_list)
        dropprob = random.choice(dropoutList)
        hidden_size = random.choice(hidden_size_list)
        optim = random.choice(optim_list)
        nummotif = random.choice(nummotif_list)
        motiflen = random.choice(motiflen_list)
        conv_layers = random.choice(conv_layers_list)
        dilation = random.choice(dilation_list)

        # 采样范围
        learning_rate = logsampler(0.0001, 0.01)
        weightDecay = logsampler(1e-5, 1e-3)

        print(f"Fold {fold_idx + 1} hyperparameters:")
        print(f"  RNN_hidden_size: {RNN_hidden_size}, Dropout: {dropprob}")
        print(f"  Hidden size: {hidden_size}, Optimizer: {optim}")
        print(f"  Num motifs: {nummotif}, Motif len: {motiflen}")
        print(f"  Conv layers: {conv_layers}, Dilation: {dilation}")
        print(f"  Learning rate: {learning_rate:.4f}")
        print(f"  Weight decay: {weightDecay:.4e}")

        model = CNNRNNNetwork(
            nummotif=nummotif,
            motiflen=motiflen,
            conv_layers=conv_layers,
            RNN_hidden_size=RNN_hidden_size,
            hidden_size=hidden_size,
            dropprob=dropprob,
            dilation=dilation,
            RNN_type=RNN_type,
            RNN_layers=RNN_layers
        ).to(device)

        # 使用优化器
        if optim == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weightDecay
            )
        elif optim == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weightDecay
            )
        elif optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weightDecay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weightDecay
            )

        # 训练模型
        best_auc = 0
        patience = 5
        patience_counter = 0

        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for i, (data1, data2, target) in enumerate(train_loader):
                data1 = data1.to(device)
                data2 = data2.to(device)
                target = target.to(device)

                # 前向传播
                output = model(data1, data2)
                loss = F.binary_cross_entropy(output, target)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches

            # 验证
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for j, (data1_val, data2_val, target_val) in enumerate(valid_loader):
                    data1_val = data1_val.to(device)
                    data2_val = data2_val.to(device)
                    target_val = target_val.to(device)

                    output = model(data1_val, data2_val)

                    pred = output.cpu().detach().numpy().reshape(output.shape[0])
                    labels = target_val.cpu().numpy().reshape(output.shape[0])

                    all_preds.extend(pred.tolist())
                    all_labels.extend(labels.tolist())

            if all_labels:
                auc_score = metrics.roc_auc_score(all_labels, all_preds)

                # 早停机制
                if auc_score > best_auc:
                    best_auc = auc_score
                    patience_counter = 0
                    # 更新全局最佳超参数
                    if best_auc > best_auc_overall:
                        best_auc_overall = best_auc
                        best_hparams = {
                            'RNN_hidden_size': RNN_hidden_size,
                            'dropprob': dropprob,
                            'hidden_size': hidden_size,
                            'optim': optim,
                            'nummotif': nummotif,
                            'motiflen': motiflen,
                            'conv_layers': conv_layers,
                            'dilation': dilation,
                            'learning_rate': learning_rate,
                            'weightDecay': weightDecay
                        }
                else:
                    patience_counter += 1

                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | Val AUC: {auc_score:.4f} | "
                      f"Best AUC: {best_auc:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | No validation samples")

        # 记录本折的最佳AUC
        fold_results.append(best_auc)
        print(f"Fold {fold_idx + 1} best AUC: {best_auc:.4f}")

    # 计算平均AUC
    mean_auc = np.mean(fold_results)
    std_auc = np.std(fold_results)
    print(f"\n5-Fold CV Calibration Results:")
    for i, auc in enumerate(fold_results):
        print(f"  Fold {i + 1}: AUC = {auc:.4f}")
    print(f"Mean AUC: {mean_auc:.4f}, Std: {std_auc:.4f}")

    # 返回最佳超参数
    best_hyperparameters = {
        'best_learning_rate': best_hparams['learning_rate'],
        'best_dropprob': best_hparams['dropprob'],
        'best_RNN_hidden_size': best_hparams['RNN_hidden_size'],
        'best_weightDecay': best_hparams['weightDecay'],
        'best_hidden_size': best_hparams['hidden_size'],
        'best_optim': best_hparams['optim'],
        'best_nummotif': best_hparams['nummotif'],
        'best_motiflen': best_hparams['motiflen'],
        'best_conv_layers': best_hparams['conv_layers'],
        'best_dilation': best_hparams['dilation']
    }

    print("Best hyperparameters from calibration:")
    for key, value in best_hyperparameters.items():
        print(f"  {key}: {value}")

    return best_hyperparameters


def find_optimal_threshold(labels, probs):
    """使用ROC曲线找到最佳分类阈值"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # 使用Youden指数找到最佳阈值
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def Train_model(folds, best_hyperparameters):
    """使用最佳超参数在五折上训练模型"""
    print('\n=== Training models with best hyperparameters using 5-fold CV ===')

    # 从最佳超参数中提取值
    best_learning_rate = best_hyperparameters['best_learning_rate']
    best_dropprob = best_hyperparameters['best_dropprob']
    best_RNN_hidden_size = best_hyperparameters['best_RNN_hidden_size']
    best_weightDecay = best_hyperparameters['best_weightDecay']
    best_hidden_size = best_hyperparameters['best_hidden_size']
    best_optim = best_hyperparameters['best_optim']
    best_nummotif = best_hyperparameters['best_nummotif']
    best_motiflen = best_hyperparameters['best_motiflen']
    best_conv_layers = best_hyperparameters['best_conv_layers']
    best_dilation = best_hyperparameters['best_dilation']

    # 训练每一折
    fold_results = []
    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\nTraining model for fold {fold_idx + 1}/5')
        print(f"Using optimizer: {best_optim}")

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data)
        valid_dataset = InteractionDatasetClass(valid_data)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = CNNRNNNetwork(
            nummotif=best_nummotif,
            motiflen=best_motiflen,
            conv_layers=best_conv_layers,
            RNN_hidden_size=best_RNN_hidden_size,
            hidden_size=best_hidden_size,
            dropprob=best_dropprob,
            dilation=best_dilation,
            RNN_type=RNN_type,
            RNN_layers=RNN_layers
        ).to(device)

        # 选择优化器
        if best_optim == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )
        elif best_optim == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )
        elif best_optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )

        # 训练参数
        num_epochs = 100
        best_val_auc = 0
        patience = 20
        patience_counter = 0

        print(f"Training for up to {num_epochs} epochs with patience {patience}...")

        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            # 训练阶段
            for i, (data1, data2, target) in enumerate(train_loader):
                data1 = data1.to(device)
                data2 = data2.to(device)
                target = target.to(device)

                # 前向传播
                output = model(data1, data2)
                loss = F.binary_cross_entropy(output, target)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches

            # 验证阶段
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for j, (data1_val, data2_val, target_val) in enumerate(valid_loader):
                    data1_val = data1_val.to(device)
                    data2_val = data2_val.to(device)
                    target_val = target_val.to(device)

                    output = model(data1_val, data2_val)

                    pred = output.cpu().detach().numpy().reshape(output.shape[0])
                    # 修正这里：将 .c.numpy() 改为 .cpu().numpy()
                    labels = target_val.cpu().numpy().reshape(output.shape[0])

                    all_preds.extend(pred.tolist())
                    all_labels.extend(labels.tolist())

            # 计算验证AUC
            if all_labels:
                val_auc = metrics.roc_auc_score(all_labels, all_preds)

                # 打印训练进度
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | Val AUC: {val_auc:.4f}")

                # 早停机制
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(model.state_dict(), f'best_model_fold{fold_idx + 1}.pth')
                    print(f"  Best model saved (AUC: {best_val_auc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | No validation samples")

        # 加载最佳模型
        model.load_state_dict(torch.load(f'best_model_fold{fold_idx + 1}.pth'))
        print(f"Loaded best model for fold {fold_idx + 1}")

        # 在验证集上评估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data1_val, data2_val, target_val in valid_loader:
                data1_val = data1_val.to(device)
                data2_val = data2_val.to(device)
                target_val = target_val.to(device)

                output = model(data1_val, data2_val)

                pred = output.cpu().detach().numpy().reshape(output.shape[0])
                # 修正这里：将 .c.numpy() 改为 .cpu().numpy()
                labels = target_val.cpu().numpy().reshape(output.shape[0])

                all_preds.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        # 计算所有指标
        if len(all_labels) > 1:
            # 计算AUC
            auc_score = metrics.roc_auc_score(all_labels, all_preds)

            # 找到最佳阈值
            optimal_threshold = find_optimal_threshold(all_labels, all_preds)
            print(f"Optimal threshold for fold {fold_idx + 1}: {optimal_threshold:.4f}")

            # 使用最佳阈值进行二分类
            all_preds_binary = [1 if p > optimal_threshold else 0 for p in all_preds]

            # 计算其他指标
            accuracy = metrics.accuracy_score(all_labels, all_preds_binary)
            precision = metrics.precision_score(all_labels, all_preds_binary)
            recall = metrics.recall_score(all_labels, all_preds_binary)
            f1 = metrics.f1_score(all_labels, all_preds_binary)
            confusion_matrix = metrics.confusion_matrix(all_labels, all_preds_binary)

            # 保存结果
            fold_results.append({
                'auc': auc_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix,
                'optimal_threshold': optimal_threshold
            })

            print(f"Fold {fold_idx + 1} Validation Results:")
            print(f"  AUC: {auc_score:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Confusion Matrix:\n{confusion_matrix}")
            print(f"  Optimal Threshold: {optimal_threshold:.4f}")

            # 保存模型
            model_path = f'model_fold{fold_idx + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Warning: Not enough samples to calculate metrics for fold {fold_idx + 1}")
            fold_results.append(None)

    # 计算平均指标
    metrics_list = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    mean_metrics = {}
    std_metrics = {}
    thresholds = []

    for metric in metrics_list:
        values = [fold[metric] for fold in fold_results if fold is not None]
        if values:
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)

    for fold in fold_results:
        if fold is not None:
            thresholds.append(fold['optimal_threshold'])

    print(f"\nFinal 5-Fold CV Results:")
    for metric in metrics_list:
        if metric in mean_metrics:
            print(f"  Mean {metric.upper()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    if thresholds:
        print(f"  Optimal thresholds: {[f'{t:.4f}' for t in thresholds]}")
        print(f"  Mean optimal threshold: {np.mean(thresholds):.4f}")

    # 保存详细结果
    with open('cross_validation_results.txt', 'w') as f:
        f.write("Detailed 5-Fold Cross Validation Results:\n\n")

        for i, result in enumerate(fold_results):
            if result is None:
                continue

            f.write(f"Fold {i + 1} Results:\n")
            f.write(f"  AUC: {result['auc']:.4f}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1 Score: {result['f1']:.4f}\n")
            f.write(f"  Optimal Threshold: {result['optimal_threshold']:.4f}\n")
            f.write(f"  Confusion Matrix:\n{result['confusion_matrix']}\n\n")

        f.write("\nOverall Metrics (Mean ± Std):\n")
        for metric in metrics_list:
            if metric in mean_metrics:
                f.write(f"  {metric.upper()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")

        f.write(f"Optimal thresholds: {[f'{t:.4f}' for t in thresholds]}\n")
        f.write(f"Mean optimal threshold: {np.mean(thresholds):.4f}\n")

    # 返回平均AUC（保持兼容性）
    return mean_metrics.get('auc', 0.0)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Sequence interaction prediction with 5-fold CV')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data in CSV format: sequence1,sequence2,label')
    parser.add_argument('--data_type', type=str, default='DNA', choices=['DNA', 'RNA'],
                        help='Type of sequence data: DNA or RNA')
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Directory to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--no-train', dest='train', action='store_false',
                        help='Skip training')
    parser.set_defaults(train=True)
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to save/load model')
    parser.add_argument('--out_file', type=str, default='results.csv',
                        help='Output file for predictions/results')

    return parser.parse_args()


def main():
    try:
        # 解析命令行参数
        args = parse_arguments()
        global data_type
        data_type = args.data_type

        # 打印参数
        print("Command line arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")

        # 确保模型目录存在
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # 加载数据并准备五折交叉验证
        print("Loading data...")
        folds = Load_Data(args.train_data)
        print(f"Loaded data with {len(folds)} folds")

        if args.train:
            print("Starting calibration and training...")
            # 超参数校准
            best_hyperparameters = Calibration(folds)

            # 使用最佳超参数训练模型
            cv_auc = Train_model(folds, best_hyperparameters)
        else:
            print("Skipping training as --train is set to False")

        print(f"\n=== Training complete ===")
        print(f"Results saved to cross_validation_results.txt")

    except Exception as e:
        print(f"\n!!! Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 在程序入口设置随机种子


    main()
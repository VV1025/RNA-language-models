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
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import tempfile
import shutil

# 设置全局随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 全局变量
bases = 'ACGT'  # DNA bases
basesRNA = 'ACGU'  # RNA bases
batch_size = 128
RNN = True
data_type = 'DNA'
kmer_size = 3  # k-mer大小
freeze_embedding = True  # 是否冻结Word2Vec嵌入层


# 序列到k-mer的转换函数
def seq_to_kmers(seq, k=kmer_size):
    """将序列转换为k-mer序列"""
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


# 序列到索引的转换函数
def kmers_to_index(kmers, w2v_model):
    """将k-mer序列转换为索引序列"""
    # 创建k-mer到索引的映射
    indices = []
    for kmer in kmers:
        if kmer in w2v_model.wv:
            indices.append(w2v_model.wv.key_to_index[kmer] + 1)
        else:
            indices.append(0)  # 0用于未知k-mer
    return np.array(indices)


def train_word2vec(all_sequences, vector_size=100, window=5, min_count=1, workers=4):
    """训练Word2Vec模型"""
    print(f"Training Word2Vec model with {len(all_sequences)} sequences...")

    # 创建临时文件保存所有k-mer序列
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "all_kmers.txt")

    with open(temp_file, 'w') as f:
        for seq in all_sequences:
            kmers = seq_to_kmers(seq)
            f.write(' '.join(kmers) + '\n')

    # 训练Word2Vec模型
    sentences = LineSentence(temp_file)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10
    )

    # 清理临时文件
    shutil.rmtree(temp_dir)

    print(f"Word2Vec model trained. Vocabulary size: {len(model.wv)}")
    return model


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


class Network(nn.Module):
    def __init__(self, embedding_dim, RNN_hidden_size, hidden_size, hidden, dropprob, sigmaRNN, xavier_init,
                 w2v_weights=None):
        super(Network, self).__init__()

        self.hidden = hidden
        self.RNN_hidden_size = RNN_hidden_size
        self.dropprob = dropprob

        # 嵌入层参数
        if w2v_weights is not None:
            vocab_size, embedding_dim = w2v_weights.shape
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0  # 填充索引为0
            )
            # 使用预训练的Word2Vec权重初始化嵌入层
            self.embedding.weight.data.copy_(torch.from_numpy(w2v_weights))
            if freeze_embedding:
                self.embedding.weight.requires_grad = False
                print("Word2Vec embeddings frozen")
            else:
                print("Word2Vec embeddings fine-tuning enabled")
        else:
            # 如果没有提供Word2Vec权重，使用随机初始化
            vocab_size = len(bases) ** kmer_size + 2  # 估计的词汇量
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0
            )
            print("Using randomly initialized embeddings")

        # RNN for sequence 1
        self.rnn1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=RNN_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # RNN for sequence 2
        self.rnn2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=RNN_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # 设置全连接层大小
        self.FC_size = 4 * RNN_hidden_size  # 双向且两个序列

        # 初始化RNN权重
        if not xavier_init:
            for name, param in self.rnn1.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0, std=sigmaRNN)
            for name, param in self.rnn2.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0, std=sigmaRNN)
        else:
            for name, param in self.rnn1.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            for name, param in self.rnn2.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)

        # FC layers
        if hidden:
            self.fc = nn.Sequential(
                nn.Linear(self.FC_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropprob),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.fc = nn.Linear(self.FC_size, 1)

        # 初始化全连接层
        if not xavier_init:
            if hidden:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.normal_(layer.weight, mean=0, std=sigmaRNN)
                        if layer.bias is not None:
                            torch.nn.init.normal_(layer.bias, mean=0, std=sigmaRNN)
            else:
                torch.nn.init.normal_(self.fc.weight, mean=0, std=sigmaRNN)
                if self.fc.bias is not None:
                    torch.nn.init.normal_(self.fc.bias, mean=0, std=sigmaRNN)
        else:
            if hidden:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
            else:
                torch.nn.init.xavier_uniform_(self.fc.weight)

        self.dropout = nn.Dropout(p=dropprob)

    def forward(self, x1, x2):
        # 嵌入层转换
        x1 = self.embedding(x1)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        x2 = self.embedding(x2)

        # 调整维度
        x1 = x1.permute(1, 0, 2)
        output1, _ = self.rnn1(x1)

        # 双向RNN: 取最后时间步的前向和后向隐藏状态
        forward1 = output1[-1, :, :self.RNN_hidden_size]
        backward1 = output1[-1, :, self.RNN_hidden_size:]
        x1 = torch.cat((forward1, backward1), dim=1)
        x1 = self.dropout(x1)

        # 处理序列2
        x2 = x2.permute(1, 0, 2)
        output2, _ = self.rnn2(x2)
        forward2 = output2[-1, :, :self.RNN_hidden_size]
        backward2 = output2[-1, :, self.RNN_hidden_size:]
        x2 = torch.cat((forward2, backward2), dim=1)
        x2 = self.dropout(x2)

        # 合并特征
        x = torch.cat((x1, x2), dim=1)

        # 全连接层
        x = self.fc(x)
        return torch.sigmoid(x)


class InteractionDataset():
    def __init__(self, filename):
        self.file = filename

    def openFile(self):
        """读取CSV文件并准备数据集"""
        dataset = []
        sequences1 = []
        sequences2 = []
        all_sequences = []  # 收集所有序列用于Word2Vec训练

        with open(self.file, 'r') as data:
            reader = csv.reader(data, delimiter=',')
            header = next(reader)  # 跳过标题行

            for row in reader:
                if len(row) < 3:
                    continue
                seq1, seq2, label = row[0], row[1], int(row[2])
                sequences1.append(seq1)
                sequences2.append(seq2)
                all_sequences.append(seq1)
                all_sequences.append(seq2)

                # 存储原始序列
                dataset.append([[seq1, seq2], [label]])

        # 添加数据统计
        num_positive = sum(item[1][0] for item in dataset)
        print(f"Loaded {len(dataset)} samples, positive: {num_positive}, negative: {len(dataset) - num_positive}")

        return dataset, sequences1, sequences2, all_sequences


def k_fold_split(dataset, k=5):
    """将数据集划分为k折用于交叉验证"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(dataset):
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        folds.append((train_data, val_data))
    return folds


class InteractionDatasetClass(Dataset):
    """处理索引序列的数据集类"""

    def __init__(self, xy=None, w2v_model=None):
        # 找到最大序列长度（在k-mer转换后）
        max_len = 0
        for el in xy:
            seq1_kmers = seq_to_kmers(el[0][0])
            seq2_kmers = seq_to_kmers(el[0][1])
            max_len = max(max_len, len(seq1_kmers), len(seq2_kmers))

        # 初始化列表存储填充后的数据
        x1_list = []
        x2_list = []

        for el in xy:
            # 对序列1进行k-mer转换和填充
            seq1 = el[0][0]
            kmers1 = seq_to_kmers(seq1)
            x1 = kmers_to_index(kmers1, w2v_model)
            pad_width = max_len - len(kmers1)
            if pad_width > 0:
                x1 = np.pad(x1, (0, pad_width), 'constant')
            x1_list.append(x1)

            # 对序列2进行k-mer转换和填充
            seq2 = el[0][1]
            kmers2 = seq_to_kmers(seq2)
            x2 = kmers_to_index(kmers2, w2v_model)
            pad_width = max_len - len(kmers2)
            if pad_width > 0:
                x2 = np.pad(x2, (0, pad_width), 'constant')
            x2_list.append(x2)

        # 转换为numpy数组
        self.x1_data = np.stack(x1_list)
        self.x2_data = np.stack(x2_list)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)

        # 转换为torch张量
        self.x1_data = torch.from_numpy(self.x1_data).long()
        self.x2_data = torch.from_numpy(self.x2_data).long()
        self.y_data = torch.from_numpy(self.y_data).float()
        self.len = len(self.x1_data)
        self.seq_len = max_len

    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def Load_Data(train_file):
    """加载数据并准备五折交叉验证"""
    print(f"Loading data from {train_file}")
    interaction_data = InteractionDataset(train_file)
    dataset, _, _, all_sequences = interaction_data.openFile()

    # 训练Word2Vec模型
    w2v_model = train_word2vec(all_sequences, vector_size=100)

    # 五折交叉验证划分
    print("Creating 5-fold cross validation splits")
    folds = k_fold_split(dataset, k=5)

    return folds, w2v_model


def Calibration(folds, w2v_model):
    """使用五折交叉验证进行超参数校准"""
    print('Starting automatic calibration with 5-fold cross validation')

    # 确保随机状态一致
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 准备Word2Vec权重
    vocab_size = len(w2v_model.wv) + 1  # +1 for padding
    w2v_weights = np.zeros((vocab_size, w2v_model.vector_size))
    for i in range(len(w2v_model.wv)):
        w2v_weights[i + 1] = w2v_model.wv.vectors[i]  # index 0 is for padding

    # 修改后的超参数搜索空间
    RNN_hidden_size_list = [64, 128, 256]
    dropoutList = [0.3, 0.5]
    hidden_list = [True]
    xavier_List = [True]
    hidden_size_list = [64, 128, 256]
    optim_list = ['Adam', 'AdamW']
    num_epochs = 30

    # 存储每折的最佳结果和超参数
    fold_results = []
    fold_hyperparams = []
    best_auc_overall = 0
    best_hparams = None

    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\n=== Processing fold {fold_idx + 1}/5 ===')

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data, w2v_model)
        valid_dataset = InteractionDatasetClass(valid_data, w2v_model)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 随机采样超参数
        RNN_hidden_size = random.choice(RNN_hidden_size_list)
        dropprob = random.choice(dropoutList)
        hidden = random.choice(hidden_list)
        xavier = random.choice(xavier_List)
        hidden_size = random.choice(hidden_size_list)
        optim = random.choice(optim_list)

        # 修改采样范围
        learning_rate = logsampler(0.001, 0.01)
        sigmaRNN = logsampler(0.01, 0.1)
        weightDecay = logsampler(1e-4, 1e-2)

        # 记录当前超参数
        current_hparams = {
            'RNN_hidden_size': RNN_hidden_size,
            'dropprob': dropprob,
            'hidden': hidden,
            'xavier': xavier,
            'hidden_size': hidden_size,
            'optim': optim,
            'learning_rate': learning_rate,
            'sigmaRNN': sigmaRNN,
            'weightDecay': weightDecay
        }

        print(f"Fold {fold_idx + 1} hyperparameters:")
        print(f"  RNN_hidden_size: {RNN_hidden_size}, Dropout: {dropprob}")
        print(f"  Hidden layers: {hidden}, Xavier init: {xavier}, Hidden size: {hidden_size}")
        print(f"  Optimizer: {optim}, Learning rate: {learning_rate:.4f}")
        print(f"  SigmaRNN: {sigmaRNN:.4e}, Weight decay: {weightDecay:.4e}")

        model = Network(
            embedding_dim=100,  # 固定为Word2Vec维度
            RNN_hidden_size=RNN_hidden_size,
            hidden_size=hidden_size,
            hidden=hidden,
            dropprob=dropprob,
            sigmaRNN=sigmaRNN,
            xavier_init=xavier,
            w2v_weights=w2v_weights
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
                else:
                    patience_counter += 1

                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | Val AUC: {auc_score:.4f} | "
                      f"Best AUC: {best_auc:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | No validation samples")

        # 记录本折的最佳AUC和超参数
        fold_best_auc = best_auc
        fold_results.append(fold_best_auc)
        fold_hyperparams.append(current_hparams)
        print(f"Fold {fold_idx + 1} best AUC: {fold_best_auc:.4f}")

        # 更新全局最佳超参数
        if fold_best_auc > best_auc_overall:
            best_auc_overall = fold_best_auc
            best_hparams = current_hparams
            print(f"New best AUC: {best_auc_overall:.4f} with optimizer: {best_hparams['optim']}")

    # 计算平均AUC
    mean_auc = np.mean(fold_results)
    std_auc = np.std(fold_results)
    print(f"\n5-Fold CV Calibration Results:")
    for i, auc in enumerate(fold_results):
        print(f"  Fold {i + 1}: AUC = {auc:.4f}")
    print(f"Mean AUC: {mean_auc:.4f}, Std: {std_auc:.4f}")

    # 返回实际测试出的最佳超参数
    best_hyperparameters = {
        'best_learning_rate': best_hparams['learning_rate'],
        'best_dropprob': best_hparams['dropprob'],
        'best_RNN_hidden_size': best_hparams['RNN_hidden_size'],
        'best_weightDecay': best_hparams['weightDecay'],
        'best_hidden': best_hparams['hidden'],
        'best_sigmaRNN': best_hparams['sigmaRNN'],
        'best_xavier': best_hparams['xavier'],
        'best_optim': best_hparams['optim'],
        'best_hidden_size': best_hparams['hidden_size']
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


def Train_model(folds, best_hyperparameters, model_dir, model_path, out_file, w2v_model):
    """使用最佳超参数在五折上训练模型"""
    print('\n=== Training models with best hyperparameters using 5-fold CV ===')

    # 打印最终使用的优化器
    final_optim = best_hyperparameters['best_optim']
    print(f"Final optimizer selected: {final_optim}")

    # 准备Word2Vec权重
    vocab_size = len(w2v_model.wv) + 1  # +1 for padding
    w2v_weights = np.zeros((vocab_size, w2v_model.vector_size))
    for i in range(len(w2v_model.wv)):
        w2v_weights[i + 1] = w2v_model.wv.vectors[i]  # index 0 is for padding

    # 确保随机状态一致
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 从最佳超参数中提取值
    best_learning_rate = best_hyperparameters['best_learning_rate']
    best_dropprob = best_hyperparameters['best_dropprob']
    best_RNN_hidden_size = best_hyperparameters['best_RNN_hidden_size']
    best_weightDecay = best_hyperparameters['best_weightDecay']
    best_hidden = best_hyperparameters['best_hidden']
    best_sigmaRNN = best_hyperparameters['best_sigmaRNN']
    best_xavier = best_hyperparameters['best_xavier']
    best_optim = best_hyperparameters['best_optim']
    best_hidden_size = best_hyperparameters['best_hidden_size']

    # 训练每一折
    fold_results = []
    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\nTraining model for fold {fold_idx + 1}/5')
        print(f"Using optimizer: {best_optim}")

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data, w2v_model)
        valid_dataset = InteractionDatasetClass(valid_data, w2v_model)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = Network(
            embedding_dim=100,  # 固定为Word2Vec维度
            RNN_hidden_size=best_RNN_hidden_size,
            hidden_size=best_hidden_size,
            hidden=best_hidden,
            dropprob=best_dropprob,
            sigmaRNN=best_sigmaRNN,
            xavier_init=best_xavier,
            w2v_weights=w2v_weights
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
        elif best_optim == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )
        else:
            print("Warning: Unrecognized optimizer, using AdamW as fallback")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=best_learning_rate,
                weight_decay=best_weightDecay
            )

        # 训练参数
        num_epochs = 200
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
                    model_save_path = os.path.join(model_dir, f"{model_path}_fold{fold_idx + 1}.pth")
                    torch.save(model.state_dict(), model_save_path)
                    print(f"  Best model saved to {model_save_path} (AUC: {best_val_auc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_epoch_loss:.4f} | No validation samples")

        # 加载最佳模型
        model_save_path = os.path.join(model_dir, f"{model_path}_fold{fold_idx + 1}.pth")
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model for fold {fold_idx + 1} from {model_save_path}")

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
                labels = target_val.cpu().numpy().reshape(output.shape[0])

                all_preds.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        # 计算所有指标
        if all_labels:
            # 计算AUC
            auc_score = metrics.roc_auc_score(all_labels, all_preds)

            # 找到最佳阈值
            optimal_threshold = find_optimal_threshold(all_labels, all_preds)

            # 计算二值化预测
            binary_preds = [1 if p >= optimal_threshold else 0 for p in all_preds]

            # 计算其他指标
            accuracy = metrics.accuracy_score(all_labels, binary_preds)
            precision = metrics.precision_score(all_labels, binary_preds)
            recall = metrics.recall_score(all_labels, binary_preds)
            f1 = metrics.f1_score(all_labels, binary_preds)
            mcc = metrics.matthews_corrcoef(all_labels, binary_preds)

            # 计算混淆矩阵
            tn, fp, fn, tp = metrics.confusion_matrix(all_labels, binary_preds).ravel()

            print(f"Fold {fold_idx + 1} Evaluation Results:")
            print(f"  AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
            print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            print(f"  Optimal Threshold: {optimal_threshold:.4f}")

            fold_results.append({
                'auc': auc_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'probs': all_preds,
                'labels': all_labels
            })
        else:
            print(f"Fold {fold_idx + 1}: Not enough samples for evaluation")
            fold_results.append(None)

    # 保存结果到CSV文件
    if out_file:
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'TP', 'TN', 'FP', 'FN'])
            for i, result in enumerate(fold_results):
                if result:
                    writer.writerow([
                        i + 1,
                        result['auc'],
                        result['accuracy'],
                        result['precision'],
                        result['recall'],
                        result['f1'],
                        result['mcc'],
                        result['tp'],
                        result['tn'],
                        result['fp'],
                        result['fn']
                    ])
        print(f"\nResults saved to {out_file}")

    # 计算平均指标
    valid_results = [r for r in fold_results if r is not None]
    if valid_results:
        avg_auc = np.mean([r['auc'] for r in valid_results])
        avg_acc = np.mean([r['accuracy'] for r in valid_results])
        avg_precision = np.mean([r['precision'] for r in valid_results])
        avg_recall = np.mean([r['recall'] for r in valid_results])
        avg_f1 = np.mean([r['f1'] for r in valid_results])
        avg_mcc = np.mean([r['mcc'] for r in valid_results])

        print("\n=== Final 5-Fold CV Results ===")
        print(f"Average AUC: {avg_auc:.4f}")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average MCC: {avg_mcc:.4f}")

        # 打印每折结果
        for i, result in enumerate(fold_results):
            if result:
                print(f"\nFold {i + 1} Results:")
                print(f"  AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
                print(f"  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")
                print(f"  MCC: {result['mcc']:.4f}")
                print(f"  Confusion Matrix: TP={result['tp']}, TN={result['tn']}, FP={result['fp']}, FN={result['fn']}")
    else:
        print("No valid results to report")

    return fold_results


def main():
    parser = argparse.ArgumentParser(description='Sequence Interaction Prediction with Word2Vec Embeddings')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--data_type', type=str, choices=['DNA', 'RNA'], default='DNA', help='Data type: DNA or RNA')
    parser.add_argument('--model_dir', type=str, default='trained_models', help='Directory to save trained models')
    parser.add_argument('--model_path', type=str, default='model', help='Base name for model files')
    parser.add_argument('--out_file', type=str, default='results.csv', help='Output file for results')
    parser.add_argument('--kmer', type=int, default=3, help='k-mer size for Word2Vec')
    parser.add_argument('--freeze_embedding', type=bool, default=True, help='Freeze Word2Vec embeddings')
    args = parser.parse_args()

    global data_type, kmer_size, freeze_embedding
    data_type = args.data_type
    kmer_size = args.kmer
    freeze_embedding = args.freeze_embedding
    print(f"Using data type: {data_type}, k-mer size: {kmer_size}, freeze embedding: {freeze_embedding}")

    # 创建模型目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 加载数据
    folds, w2v_model = Load_Data(args.train_data)

    # 超参数校准
    best_hyperparameters = Calibration(folds, w2v_model)

    # 使用最佳超参数训练模型
    Train_model(folds, best_hyperparameters, args.model_dir, args.model_path, args.out_file, w2v_model)

    # 保存Word2Vec模型
    w2v_model.save(os.path.join(args.model_dir, "word2vec.model"))
    print("Word2Vec model saved")

    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
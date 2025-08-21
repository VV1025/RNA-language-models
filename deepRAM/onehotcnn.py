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

# 设置全局随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 全局变量
bases = 'ACGT'  # DNA bases
basesRNA = 'ACGU'  # RNA bases
batch_size = 128
data_type = 'RNA'


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


class Network(nn.Module):
    def __init__(self, input_size, conv_kernel_size, conv_out_channels, hidden_size, hidden, dropprob, sigmaCNN,
                 xavier_init):
        super(Network, self).__init__()

        self.hidden = hidden
        self.conv_out_channels = conv_out_channels
        self.dropprob = dropprob
        self.conv_kernel_size = conv_kernel_size
        self.sigmaCNN = sigmaCNN
        self.xavier_init = xavier_init
        self.hidden_size = hidden_size

        # 输入特征大小 (4 for one-hot encoding)
        self.input_size = input_size

        # 卷积层用于序列1
        self.conv1_seq1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding='same'
        )

        # 卷积层用于序列2
        self.conv1_seq2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding='same'
        )

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 展平层
        self.flatten = nn.Flatten()

        # 初始化卷积权重
        if not xavier_init:
            torch.nn.init.normal_(self.conv1_seq1.weight, mean=0, std=sigmaCNN)
            torch.nn.init.normal_(self.conv1_seq2.weight, mean=0, std=sigmaCNN)
        else:
            torch.nn.init.xavier_uniform_(self.conv1_seq1.weight)
            torch.nn.init.xavier_uniform_(self.conv1_seq2.weight)

        # FC layers - 初始化为None，将在首次前向传播时创建
        self.fc = None

        # 用于存储展平大小
        self.flatten_size = None

        self.dropout = nn.Dropout(p=dropprob)

    def _create_fc_layers(self, flatten_size):
        """根据展平特征大小创建全连接层"""
        if self.hidden:
            fc = nn.Sequential(
                nn.Linear(flatten_size * 2, self.hidden_size),  # *2是因为两个序列
                nn.ReLU(),
                nn.Dropout(self.dropprob),
                nn.Linear(self.hidden_size, 1)
            )
        else:
            fc = nn.Linear(flatten_size * 2, 1)

        # 初始化全连接层
        if not self.xavier_init:
            if self.hidden:
                for layer in fc:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.normal_(layer.weight, mean=0, std=self.sigmaCNN)
                        if layer.bias is not None:
                            torch.nn.init.normal_(layer.bias, mean=0, std=self.sigmaCNN)
            else:
                torch.nn.init.normal_(fc.weight, mean=0, std=self.sigmaCNN)
                if fc.bias is not None:
                    torch.nn.init.normal_(fc.bias, mean=0, std=self.sigmaCNN)
        else:
            if self.hidden:
                for layer in fc:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
            else:
                torch.nn.init.xavier_uniform_(fc.weight)

        return fc

    def forward(self, x1, x2):
        # 调整维度: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        # 序列1的卷积处理
        x1 = self.conv1_seq1(x1)
        x1 = F.relu(x1)
        x1 = self.pool(x1)
        x1 = self.flatten(x1)

        # 序列2的卷积处理
        x2 = self.conv1_seq2(x2)
        x2 = F.relu(x2)
        x2 = self.pool(x2)
        x2 = self.flatten(x2)

        # 确定展平特征大小并创建全连接层
        if self.flatten_size is None:
            self.flatten_size = x1.size(1)
            self.fc = self._create_fc_layers(self.flatten_size).to(device)

        # 合并特征
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)

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
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
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
            x2 = el[0][1]
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

    # 确保随机状态一致
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # CNN架构的超参数搜索空间
    conv_kernel_size_list = [3, 5, 7]  # 卷积核大小
    conv_out_channels_list = [64, 128, 256]  # 卷积输出通道数
    dropoutList = [0.3, 0.5]  # dropout概率
    hidden_list = [True]  # 强制使用隐藏层
    xavier_List = [True]  # 强制使用Xavier初始化
    hidden_size_list = [64, 128]  # 隐藏层大小
    optim_list = ['Adam', 'AdamW']  # 优化器
    num_epochs = 30  # 训练轮数

    # 存储每折的最佳结果
    fold_results = []

    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\n=== Processing fold {fold_idx + 1}/5 ===')

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data)
        valid_dataset = InteractionDatasetClass(valid_data)

        # 获取输入特征大小
        input_size = train_dataset.input_size

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 随机采样超参数
        conv_kernel_size = random.choice(conv_kernel_size_list)
        conv_out_channels = random.choice(conv_out_channels_list)
        dropprob = random.choice(dropoutList)
        hidden = random.choice(hidden_list)
        xavier = random.choice(xavier_List)
        hidden_size = random.choice(hidden_size_list)
        optim = random.choice(optim_list)

        # 修改采样范围
        learning_rate = logsampler(0.001, 0.01)
        sigmaCNN = logsampler(0.01, 0.1)  # CNN初始化范围
        weightDecay = logsampler(1e-4, 1e-2)  # 权重衰减

        print(f"Fold {fold_idx + 1} hyperparameters:")
        print(f"  Conv kernel size: {conv_kernel_size}, Conv out channels: {conv_out_channels}")
        print(f"  Dropout: {dropprob}, Hidden layers: {hidden}")
        print(f"  Xavier init: {xavier}, Hidden size: {hidden_size}, Optimizer: {optim}")
        print(f"  Learning rate: {learning_rate:.4f}")
        print(f"  SigmaCNN: {sigmaCNN:.4e}")
        print(f"  Weight decay: {weightDecay:.4e}")

        model = Network(
            input_size=input_size,
            conv_kernel_size=conv_kernel_size,
            conv_out_channels=conv_out_channels,
            hidden_size=hidden_size,
            hidden=hidden,
            dropprob=dropprob,
            sigmaCNN=sigmaCNN,
            xavier_init=xavier
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
        fold_best_auc = best_auc
        fold_results.append(fold_best_auc)
        print(f"Fold {fold_idx + 1} best AUC: {fold_best_auc:.4f}")

    # 计算平均AUC
    mean_auc = np.mean(fold_results)
    std_auc = np.std(fold_results)
    print(f"\n5-Fold CV Calibration Results:")
    for i, auc_score in enumerate(fold_results):
        print(f"  Fold {i + 1}: AUC = {auc_score:.4f}")
    print(f"Mean AUC: {mean_auc:.4f}, Std: {std_auc:.4f}")

    # 返回最佳超参数
    best_hyperparameters = {
        'best_learning_rate': 0.01,
        'best_dropprob': 0.3,
        'best_conv_out_channels': 128,
        'best_conv_kernel_size': 5,
        'best_weightDecay': 1e-3,
        'best_hidden': True,
        'best_sigmaCNN': 0.01,
        'best_xavier': True,
        'best_optim': 'AdamW',
        'best_hidden_size': 128
    }

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

    # 确保随机状态一致
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 从最佳超参数中提取值
    best_learning_rate = best_hyperparameters['best_learning_rate']
    best_dropprob = best_hyperparameters['best_dropprob']
    best_conv_out_channels = best_hyperparameters['best_conv_out_channels']
    best_conv_kernel_size = best_hyperparameters['best_conv_kernel_size']
    best_weightDecay = best_hyperparameters['best_weightDecay']
    best_hidden = best_hyperparameters['best_hidden']
    best_sigmaCNN = best_hyperparameters['best_sigmaCNN']
    best_xavier = best_hyperparameters['best_xavier']
    best_optim = best_hyperparameters['best_optim']
    best_hidden_size = best_hyperparameters['best_hidden_size']

    # 训练每一折
    fold_results = []
    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f'\nTraining model for fold {fold_idx + 1}/5')

        # 创建DataLoader
        train_dataset = InteractionDatasetClass(train_data)
        valid_dataset = InteractionDatasetClass(valid_data)

        # 获取输入特征大小
        input_size = train_dataset.input_size

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = Network(
            input_size=input_size,
            conv_kernel_size=best_conv_kernel_size,
            conv_out_channels=best_conv_out_channels,
            hidden_size=best_hidden_size,
            hidden=best_hidden,
            dropprob=best_dropprob,
            sigmaCNN=best_sigmaCNN,
            xavier_init=best_xavier
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
            precision = metrics.precision_score(all_labels, all_preds_binary, zero_division=0)
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

    # 返回平均AUC
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
    parser.add_argument('--evaluate_performance', action='store_true',
                        help='Whether to evaluate model performance')
    parser.add_argument('--no-evaluate_performance', dest='evaluate_performance', action='store_false',
                        help='Skip evaluation')
    parser.set_defaults(evaluate_performance=True)
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to save/load model')
    parser.add_argument('--out_file', type=str, default='results.csv',
                        help='Output file for predictions/results')
    # CNN参数
    parser.add_argument('--conv_kernel_size', type=int, default=5,
                        help='Convolution kernel size')
    parser.add_argument('--conv_out_channels', type=int, default=128,
                        help='Number of output channels in convolution layer')
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'RMSprop', 'SGD'],
                        help='Optimizer type: Adam, AdamW, RMSprop, SGD')

    return parser.parse_args()


def main():
    global data_type
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 打印参数
        print("=" * 50)
        print("Command line arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        print("=" * 50)

        # 更新全局参数
        data_type = args.data_type

        print("\nUpdated global configuration:")
        print(f"  data_type: {data_type}")
        print("-" * 50)

        # 确保模型目录存在
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
            print(f"Created model directory: {args.model_dir}")

        # 加载数据并准备五折交叉验证
        print("\nLoading data...")
        folds = Load_Data(args.train_data)
        print(f"Loaded data with {len(folds)} folds")
        if folds:
            print(f"First fold contains {len(folds[0][0])} training samples and {len(folds[0][1])} validation samples")
        print("-" * 50)

        if args.train:
            print("\nStarting calibration and training...")
            start_time = time.time()

            # 超参数校准
            print("\n=== Hyperparameter Calibration ===")
            best_hyperparameters = Calibration(folds)

            # 使用命令行参数覆盖优化器选择
            if hasattr(args, 'optimizer'):
                best_hyperparameters['best_optim'] = args.optimizer
                print(f"Using optimizer from command line: {args.optimizer}")
            else:
                print(f"Using optimizer from calibration: {best_hyperparameters['best_optim']}")

            print("\nBest hyperparameters:")
            for key, value in best_hyperparameters.items():
                print(f"  {key}: {value}")
            print("-" * 50)

            # 使用最佳超参数训练模型
            print("\n=== Model Training ===")
            cv_auc = Train_model(folds, best_hyperparameters)

            end_time = time.time()
            print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
        else:
            print("\nSkipping training as --train is set to False")

        print(f"\n{'-' * 50}")
        print(f"=== Training complete ===")
        print(f"Results saved to cross_validation_results.txt")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"\n{'!' * 50}")
        print(f"!!! Error occurred: {str(e)}")
        print(f"{'!' * 50}")
        import traceback
        traceback.print_exc()
        with open('error_log.txt', 'w') as log_file:
            log_file.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Error message: {str(e)}\n")
            log_file.write("Traceback:\n")
            traceback.print_exc(file=log_file)


if __name__ == "__main__":
    # 在程序入口设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main()
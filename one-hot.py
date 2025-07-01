import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_score, f1_score, \
    recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


# ---------------------------- 原有工具函数保持不变 ----------------------------
def one_hot_coding(base):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    return mapping.get(base, [0, 0, 0, 0])


def one_hot_sequence(sequence, max_len=7):
    """将RNA序列转换为one-hot编码（自动中心填充至7个碱基）"""
    feature = []
    pad_left = (max_len - len(sequence)) // 2
    pad_right = max_len - len(sequence) - pad_left

    for _ in range(pad_left):
        feature.extend([0] * 4)

    for base in sequence[:max_len]:
        feature.extend(one_hot_coding(base))

    for _ in range(pad_right):
        feature.extend([0] * 4)

    return feature[:4 * max_len]


def generate_features(sequence, samples):
    """生成特征向量（已集成中心填充）"""
    feature_list = []
    for _, row in samples.iterrows():
        seq1 = row['Seq1']
        seq2 = row['Seq2']
        if not isinstance(seq1, str) or not isinstance(seq2, str):
            continue

        coding1 = one_hot_sequence(seq1)
        coding2 = one_hot_sequence(seq2)

        combined = np.array(coding1) + np.array(coding2)
        abs_difference = np.abs(np.array(coding1) - np.array(coding2))

        feature_vector = np.concatenate((combined, abs_difference))
        feature_list.append(feature_vector)

    return feature_list


def read_sequence_from_fasta(file_path, rna_name):
    """从FASTA文件读取指定的RNA序列"""
    sequences = {}
    with open(file_path, 'r') as file:
        sequence_name = None
        sequence_data = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_name:
                    sequences[sequence_name] = ''.join(sequence_data)
                    sequence_data = []
                sequence_name = line[1:]
            else:
                sequence_data.append(line)
        if sequence_name:
            sequences[sequence_name] = ''.join(sequence_data)
    return sequences.get(rna_name)


def read_samples_from_csv(file_path):
    """从CSV文件读取正负样本数据"""
    return pd.read_csv(file_path)


# ---------------------------- 修改后的分类器函数 ----------------------------
def classify_with_rf(X, y, output_file):
    """返回ROC数据"""
    # 数据平衡处理
    smote = SMOTE(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_balanced, y_balanced = under_sampler.fit_resample(X_resampled, y_resampled)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # 模型配置
    rf = RandomForestClassifier(
        n_estimators=1775, max_depth=20,
        min_samples_split=5, min_samples_leaf=1,
        max_features='sqrt', random_state=42
    )

    # 交叉验证预测
    y_proba = cross_val_predict(
        rf, X_scaled, y_balanced,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='predict_proba'
    )[:, 1]

    # 计算ROC
    fpr, tpr, _ = roc_curve(y_balanced, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def classify_with_svm(X, y, output_file):
    """返回ROC数据"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 模型配置
    svm = SVC(probability=True, random_state=42)

    # 交叉验证预测
    y_proba = cross_val_predict(
        svm, X_scaled, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='predict_proba'
    )[:, 1]

    # 计算ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def classify_with_knn(X, y, output_file):
    """返回ROC数据"""
    # 数据平衡处理
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # 模型配置
    knn = KNeighborsClassifier(
        n_neighbors=11,
        weights='distance',
        algorithm='ball_tree'
    )

    # 交叉验证预测
    y_proba = cross_val_predict(
        knn, X_scaled, y_resampled,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='predict_proba'
    )[:, 1]

    # 计算ROC
    fpr, tpr, _ = roc_curve(y_resampled, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# ---------------------------- 主程序流程 ----------------------------
if __name__ == '__main__':
    # 配置参数
    config = {
        'fasta_path': '/home/shenweiwei/sras/WuHan_Hu_1.fasta',
        'pos_csv': '/home/shenweiwei/sras/positive_sras_pairs_information.csv',
        'neg_csv': '/home/shenweiwei/sras/negative_sras_pairs_information.csv',
        'output_file': '/home/shenweiwei/sras/combined_roc.pdf',
        'target_sequence_name': 'NC_045512.2 Severe acute respiratory syndrome coronavirus 2 isolate Wuhan-Hu-1, complete genome'
    }

    # 加载数据
    print("正在加载数据...")
    ref_seq = read_sequence_from_fasta(config['fasta_path'], config['target_sequence_name'])
    pos_samples = read_samples_from_csv(config['pos_csv'])
    neg_samples = read_samples_from_csv(config['neg_csv'])

    # 生成特征
    print("生成特征向量...")
    X_pos = generate_features(ref_seq, pos_samples)
    X_neg = generate_features(ref_seq, neg_samples)
    X = np.array(X_pos + X_neg)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    # 运行分类器并收集ROC数据
    print("正在训练模型...")
    rf_fpr, rf_tpr, rf_auc = classify_with_rf(X, y, 'rf_report.txt')
    svm_fpr, svm_tpr, svm_auc = classify_with_svm(X, y, 'svm_report.txt')
    knn_fpr, knn_tpr, knn_auc = classify_with_knn(X, y, 'knn_report.txt')

    # 绘制综合ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(rf_fpr, rf_tpr, linestyle='-', lw=2,  # Solid line for Random Forest
             label=f'Random Forest (AUC = {rf_auc:.4f})')
    plt.plot(svm_fpr, svm_tpr, linestyle='--', lw=2,  # Dashed line for SVM
             label=f'SVM (AUC = {svm_auc:.4f})')
    plt.plot(knn_fpr, knn_tpr, linestyle='-.', lw=2,  # Dash-dot line for KNN
             label=f'KNN (AUC = {knn_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存为PDF
    plt.savefig(config['output_file'], format='pdf', dpi=800, bbox_inches='tight')
    print(f"结果已保存至 {config['output_file']}")
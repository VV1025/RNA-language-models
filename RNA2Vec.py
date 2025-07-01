import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, \
    f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from gensim.models import Word2Vec
import matplotlib.colors as mcolors


# --------------------- 数据预处理函数 ---------------------
def read_fasta(file_paths):
    """读取多个FASTA文件并返回序列字典"""
    sequences = {}
    for file_path in file_paths:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Input must be a non-empty string.")
        with open(file_path, 'r') as file:
            sequence_name = ""
            sequence_data = []
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if sequence_name:
                        sequences[sequence_name] = ''.join(sequence_data)
                    sequence_name = line[1:]
                    sequence_data = []
                else:
                    sequence_data.append(line)
            if sequence_name:
                sequences[sequence_name] = ''.join(sequence_data)
    return sequences


def k_mer(sequence, k=3, stride=1):
    """生成k-mer滑动窗口序列"""
    return [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, stride)]


def train_rna2vec(sequences, k=3):
    """训练Word2Vec模型生成嵌入向量"""
    data = [k_mer(seq, k) for seq in sequences.values()]
    model = Word2Vec(sentences=data, vector_size=50, window=5, min_count=1, workers=4, sg=1, epochs=300)
    return model


def average_elements(lists):
    """计算嵌入向量的平均值"""
    return np.mean(lists, axis=0) if lists else np.zeros(50)


def word_embedding(model, df):
    """生成每对RNA序列的特征向量（和与差拼接）"""
    features = []
    for _, row in df.iterrows():
        seq1, seq2 = row.iloc[3], row.iloc[4]
        if pd.isna(seq1) or pd.isna(seq2):
            continue

        # 提取k-mer并转换为嵌入向量
        seq1_vecs = [model.wv[word] for word in k_mer(seq1) if word in model.wv]
        seq2_vecs = [model.wv[word] for word in k_mer(seq2) if word in model.wv]

        if not seq1_vecs or not seq2_vecs:
            continue

        # 计算平均向量并拼接特征
        avg1 = average_elements(seq1_vecs)
        avg2 = average_elements(seq2_vecs)
        feature = np.concatenate([avg1 + avg2, np.abs(avg1 - avg2)])
        features.append(feature)

    return np.array(features)


# --------------------- 分类器函数（修改版） ---------------------
def RF_with_cv(X, y):
    """随机森林交叉验证返回预测概率和标签"""
    rf = RandomForestClassifier(n_estimators=1750, max_depth=20, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(rf, X, y, cv=cv)
    return y_proba, y_pred


def KNN_with_cv(X, y):
    """KNN使用固定最优参数交叉验证"""
    knn = KNeighborsClassifier(n_neighbors=11, weights='distance', algorithm='ball_tree')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(knn, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(knn, X, y, cv=cv)
    return y_proba, y_pred


def SVM_with_cv(X, y):
    """SVM交叉验证返回预测概率和标签"""
    svm = SVC(probability=True, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(svm, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(svm, X, y, cv=cv)
    return y_proba, y_pred


# --------------------- 混淆矩阵可视化函数（PDF输出） ---------------------
def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """
    绘制并保存混淆矩阵热力图为PDF格式 (800dpi)
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    # 计算百分比矩阵
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建带文本标注的热力图（调整大小适应PDF）
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 使用自定义颜色映射
    colors = ["#2E8B57", "#98FB98", "#FFA07A", "#CD5C5C"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # 绘制热力图
    sns.heatmap(cm_percent, annot=False, fmt=".1f", cmap=cmap,
                linewidths=0.5, linecolor='gray', cbar=True,
                vmin=0, vmax=100, ax=ax)

    # 添加数值标签（增加字体大小适应PDF）
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            plt.text(j + 0.5, i + 0.5,
                     f"{cm_percent[i, j]:.1f}%\n({cm[i, j]})",
                     ha='center', va='center', color=color, fontsize=12)

    # 设置标题和标签（增加字体大小）
    plt.title(f'{model_name} 混淆矩阵\n(总样本数: {total})', fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)

    # 设置坐标轴刻度
    tick_labels = ['负类 (0)', '正类 (1)']
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_yticklabels(tick_labels, fontsize=12)

    # 添加边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # 添加额外信息（增加字体大小）
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.5, -0.05,
                f"准确率: {accuracy:.4f} | 真负率: {tn / (tn + fp):.4f} | 真正率: {tp / (tp + fn):.4f}",
                ha="center", fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存为PDF（800dpi）
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.pdf")
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存至: {output_path} (800dpi PDF)")
    return output_path


# --------------------- 模型评估函数 ---------------------
def evaluate_model(model_name, y_true, y_pred, y_proba, output_dir):
    """评估模型并返回指标字典"""
    # 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 绘制混淆矩阵（PDF格式）
    cm_path = plot_confusion_matrix(y_true, y_pred, model_name, output_dir)

    # 打印指标
    print(f"\n{'-' * 60}")
    print(f"模型评估: {model_name}")
    print(f"{'-' * 60}")
    print(f"AUC值        : {auc_score:.4f}")
    print(f"准确率 (ACC)  : {acc:.4f}")
    print(f"F1值         : {f1:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确度 (Precision): {precision:.4f}")

    print("\n混淆矩阵:")
    print(f"         预测负类    预测正类")
    print(f"实际负类  {tn:>8}    {fp:>8}    (实际负类总数: {tn + fp})")
    print(f"实际正类  {fn:>8}    {tp:>8}    (实际正类总数: {fn + tp})")
    print(f"          (预测负类总数: {tn + fn})    (预测正类总数: {fp + tp})")

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['负类', '正类'], digits=4))
    print(f"{'-' * 60}\n")

    # 返回结果
    return {
        'model': model_name,
        'auc': auc_score,
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'confusion_matrix_path': cm_path
    }


# --------------------- ROC曲线绘制与保存 ---------------------
def plot_roc_curves(y_true, rf_proba, svm_proba, knn_proba, output_dir):
    """绘制并保存ROC曲线对比图为PDF (800dpi)"""
    plt.figure(figsize=(10, 8))

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 计算各模型的FPR和TPR
    models = {
        'Random Forest': rf_proba,
        'SVM': svm_proba,
        'KNN': knn_proba
    }

    # 设置颜色和线型
    colors = ['#1E90FF', '#FF6347', '#32CD32']
    linestyles = ['-', '--', '-.']

    for (name, proba), color, ls in zip(models.items(), colors, linestyles):
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                 label=f'{name} (AUC = {roc_auc:.4f})')

    # 绘制参考线
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正类率 (FPR)', fontsize=14)
    plt.ylabel('真正类率 (TPR)', fontsize=14)
    plt.title('ROC曲线对比', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)

    # 添加背景网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存为PDF (800dpi)
    output_path = os.path.join(output_dir, "roc_curves_comparison.pdf")
    plt.savefig(output_path, format='pdf', dpi=800, bbox_inches='tight')
    plt.close()

    print(f"ROC曲线对比图已保存至: {output_path} (800dpi PDF)")
    return output_path


# --------------------- 主程序 ---------------------
if __name__ == '__main__':
    # 文件路径配置
    fasta_paths = [r'/home/shenweiwei/sras/WuHan_Hu_1.fasta']
    positive_csv = r'/home/shenweiwei/sras/positive_sras_pairs_information.csv'
    negative_csv = r'/home/shenweiwei/sras/negative_sras_pairs_information.csv'
    output_dir = "/home/shenweiwei/sras/output"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有模型结果
    all_results = []

    try:
        # 数据加载与特征生成
        print("=" * 70)
        print("开始执行RNA相互作用预测模型")
        print("=" * 70)
        print("\n[步骤1/6] 正在读取FASTA文件...")
        sequences = read_fasta(fasta_paths)

        print("\n[步骤2/6] 训练Word2Vec模型生成嵌入向量...")
        model = train_rna2vec(sequences, k=3)

        print("\n[步骤3/6] 加载正负样本数据...")
        positive_df = pd.read_csv(positive_csv)
        negative_df = pd.read_csv(negative_csv)

        print("\n[步骤4/6] 生成特征向量...")
        positive_X = word_embedding(model, positive_df)
        negative_X = word_embedding(model, negative_df)

        X = np.vstack((positive_X, negative_X))
        y = np.array([1] * len(positive_X) + [0] * len(negative_X))

        print(f"\n数据集信息:")
        print(f"  总样本数: {len(y)}")
        print(f"  正样本数: {len(positive_X)} ({len(positive_X) / len(y) * 100:.1f}%)")
        print(f"  负样本数: {len(negative_X)} ({len(negative_X) / len(y) * 100:.1f}%)")
        print(f"  特征维度: {X.shape[1]}")

        # 获取各模型预测概率和标签
        print("\n[步骤5/6] 训练和评估模型...")

        # 随机森林模型
        print("\n  > 随机森林模型训练中...")
        rf_proba, rf_pred = RF_with_cv(X, y)
        rf_result = evaluate_model("Random Forest", y, rf_pred, rf_proba, output_dir)
        all_results.append(rf_result)

        # SVM模型
        print("\n  > SVM模型训练中...")
        svm_proba, svm_pred = SVM_with_cv(X, y)
        svm_result = evaluate_model("SVM", y, svm_pred, svm_proba, output_dir)
        all_results.append(svm_result)

        # KNN模型
        print("\n  > KNN模型训练中...")
        knn_proba, knn_pred = KNN_with_cv(X, y)
        knn_result = evaluate_model("KNN", y, knn_pred, knn_proba, output_dir)
        all_results.append(knn_result)

        # 绘制并保存ROC曲线
        print("\n[步骤6/6] 绘制ROC曲线...")
        roc_path = plot_roc_curves(y, rf_proba, svm_proba, knn_proba, output_dir)

        # 创建综合结果表格
        results_df = pd.DataFrame(all_results)
        results_df = results_df[['model', 'auc', 'accuracy', 'f1', 'recall', 'precision',
                                 'tp', 'fp', 'fn', 'tn', 'confusion_matrix_path']]

        # 保存结果到CSV
        results_path = os.path.join(output_dir, "model_evaluation_results.csv")
        results_df.to_csv(results_path, index=False)

        # 打印总结报告
        print("\n" + "=" * 70)
        print("模型评估总结报告")
        print("=" * 70)
        print(results_df.drop(columns=['confusion_matrix_path']).to_string(index=False))

        print(f"\n所有结果已保存至: {output_dir}")
        print(f"  1. 模型评估结果: {results_path}")
        print(f"  2. ROC曲线对比图: {roc_path}")
        print(f"  3. 混淆矩阵: 见各模型对应PDF文件")
        print("\n任务完成!")

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()
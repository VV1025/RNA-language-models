import time
import os.path as osp
from collections import defaultdict
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import paddle
import paddle.nn as nn
from paddlenlp.data import Stack
from paddle.io import Dataset
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieModel
from paddlenlp.datasets import MapDataset
import argparse
from arg_utils import (
    default_logdir,
    str2bool,
    set_seed,
    print_config,
    str2list
)

from tokenizer_nuc import NUCTokenizer
from visualizer import Visualizer
from base_classes import BaseMetrics, BaseTrainer, MlpProjector
import time
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddlenlp.data import Stack
from paddle.io import Dataset

from dataset_utils import seq2input_ids
from base_classes import BaseMetrics, BaseTrainer, MlpProjector

import time
import os.path as osp
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

import paddle
import paddle.nn as nn
from paddlenlp.data import Stack
from paddle.io import Dataset

from dataset_utils import seq2input_ids  # 假设这个包是可用的
from base_classes import BaseMetrics, BaseTrainer, MlpProjector


class GenerateRRInterTrainTest:
    """Generate train and test datasets for RNA-RNA interaction prediction."""

    def __init__(self, rr_dir, dataset, seed=0):
        """Init function."""
        csv_path = osp.join(osp.join(rr_dir, dataset), dataset + ".csv")
        self.data = pd.read_csv(csv_path, sep=",").values.tolist()
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def get(self):
        """Get the dataset."""
        return self.data


class RRInstance(object):
    """A single fine-tuning instance for classification task."""

    def __init__(self, name, tokens, input_ids, label):
        """Init function."""
        self.name = name
        self.tokens = tokens
        self.input_ids = input_ids
        self.label = label

    def __call__(self):
        """Call function."""
        return vars(self).items()


class RRInterDataset(Dataset):
    """RNA-RNA interaction dataset."""

    def __init__(self, data, tokenizer, max_seq_lens):
        """Init function."""
        super().__init__()
        self.data = [
            convert_instance_to_rr(instance, tokenizer, max_seq_lens) for instance in data
        ]

    def __getitem__(self, idx):
        """Get item."""
        instance = self.data[idx]
        return {
            "a_name": instance.name,
            "tokens": instance.tokens,  # 包含 token 的列表
            "input_ids": instance.input_ids,  # 值应该是 input_ids 列表
            "label": instance.label[0],  # label 应该是从 RRInstance 中获取
        }

    def __len__(self):
        """Get length of dataset."""
        return len(self.data)


class RRInterTrainer:
    """RNA-RNA interaction trainer."""

    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = RRInterCriterionWrapper()
        self.optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=self.args.lr)

    def cross_validate(self, num_folds=5):
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.args.seed)
        all_metrics = []

        for fold_idx, (train_index, val_index) in enumerate(kf.split(self.train_dataset.data)):
            print(f"Fold {fold_idx + 1}/{num_folds}")

            train_data = [self.train_dataset.data[i] for i in train_index]
            val_data = [self.train_dataset.data[i] for i in val_index]

            train_dataset = RRInterDataset(train_data, self.tokenizer, self.args.max_seq_lens)
            val_dataset = RRInterDataset(val_data, self.tokenizer, self.args.max_seq_lens)

            self.train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.eval_dataloader = paddle.io.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

            for epoch in range(self.args.num_train_epochs):
                self.train(epoch)
            metrics = self.eval(epoch)
            all_metrics.append(metrics)

        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        print("Cross-validation results: {}".format(avg_metrics))

    def train(self, epoch):
        """Train function."""
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=False) as pbar:
            for i, data in enumerate(self.train_dataloader):
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["label"]

                preds = self.model(tokens, input_ids)
                loss = self.loss_fn(preds, labels)

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.4f}'.format(loss_total / num_total))
                pbar.update(self.args.batch_size)

        time_ed = time.time()
        training_time = time_ed - time_st
        return training_time

    def eval(self, epoch):
        """Eval function."""
        self.model.eval()
        time_st = time.time()

        with tqdm(total=len(self.eval_dataset), disable=True) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["label"]

                with paddle.no_grad():
                    output = self.model(tokens, input_ids)

                outputs_dataset.append(output)
                labels_dataset.append(labels)
                pbar.update(self.args.batch_size)

            outputs_dataset = paddle.concat(outputs_dataset, axis=0)
            labels_dataset = paddle.concat(labels_dataset, axis=0)

            try:
                # 检查数据格式
                print(f"Model outputs shape: {outputs_dataset.shape}")
                print(f"Labels shape: {labels_dataset.shape}")
                print(f"Labels unique values: {np.unique(labels_dataset.numpy())}")

                # 计算 ROC 曲线
                probs = paddle.nn.functional.softmax(outputs_dataset, axis=-1)[:, 1].numpy()
                labels = labels_dataset.numpy()
                print(f"Probabilities (min, max): ({probs.min()}, {probs.max()})")

                fpr, tpr, thresholds = roc_curve(labels, probs)
                roc_auc = auc(fpr, tpr)
                print(f"AUC: {roc_auc}")

                # 确保输出目录存在
                os.makedirs(self.args.output, exist_ok=True)
                print(f"Output directory: {self.args.output} (exists: {os.path.exists(self.args.output)})")

                # 绘制并保存 ROC 曲线
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")

                roc_curve_path = osp.join(self.args.output, 'roc_curve_eval.png')
                plt.savefig(roc_curve_path)
                plt.close()
                print(f"ROC curve saved to: {roc_curve_path}")

            except Exception as e:
                print(f"Error in evaluation: {str(e)}")

            # 计算其他指标
            metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)

        results = {}
        log = 'Epoch {} - '.format(epoch + 1)
        for k, v in metrics_dataset.items():
            log += f"{k}: {v:.4f}\t"
            results[k] = v

        time_ed = time.time() - time_st
        print(log + "; Time: {:.4f}s".format(time_ed))

        return metrics_dataset

def convert_instance_to_rr(raw_data, tokenizer, max_seq_lens):
    """Convert raw data to RNA-RNA interaction instance."""
    a_name = raw_data[0]  # assuming raw_data follows a certain structure
    a_seq = raw_data[1]
    b_name = raw_data[2]
    b_seq = raw_data[3]
    label = raw_data[4]

    b_max_seq_length = max_seq_lens[1]
    encoder = dict(zip('NATCG', range(5)))
    tokens_a = [encoder[x] for x in a_seq.upper()]
    tokens_b = [encoder[x] for x in b_seq.upper()]

    if len(tokens_b) > b_max_seq_length:
        tokens_b = tokens_b[:b_max_seq_length]
    elif len(tokens_b) < b_max_seq_length:
        tokens_b += [0] * (b_max_seq_length - len(tokens_b))

    a_input_ids = seq2input_ids(a_seq, tokenizer)
    a_input_ids = a_input_ids[1:-1]

    b_input_ids = seq2input_ids(b_seq, tokenizer)
    b_input_ids = b_input_ids[1:-1]

    if len(b_input_ids) > b_max_seq_length:
        b_input_ids = b_input_ids[:b_max_seq_length]
    elif len(b_input_ids) < b_max_seq_length:
        b_input_ids += [0] * (b_max_seq_length - len(b_input_ids))

    name = a_name + "+" + b_name
    tokens = tokens_a + tokens_b
    input_ids = a_input_ids + b_input_ids
    label = [label]

    return RRInstance(name, tokens, input_ids, label)


class RRInterCriterionWrapper(paddle.nn.Layer):
    """Wrap criterion."""

    def __init__(self):
        """CriterionWrapper."""
        super(RRInterCriterionWrapper, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, labels):
        """Forward function."""
        labels = paddle.cast(labels, dtype='int64')
        loss = self.loss_fn(output, labels)
        return loss


# ========== Main Code Execution
if __name__ == "__main__":
    # 配置解析和设置
    parser = argparse.ArgumentParser('Implementation of RNA-RNA Interaction prediction.')

    # 模型参数
    parser.add_argument('--model_name_or_path', type=str, default="./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final/")
    parser.add_argument('--with_pretrain', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--proj_size', type=int, default=64)
    parser.add_argument('--model_path', type=str,
                        default="./output_ft/rr_inter/MirTarRAW/BERT,ERNIE,MOTIF,PROMPT/model_state.pdparams")

    # 数据参数
    parser.add_argument('--dataset', type=str, default="MirTarRAW", choices=["combined_samples"])
    parser.add_argument('--dataset_dir', type=str, default="./data/ft/rr/")
    parser.add_argument('--max_seq_lens', type=int, nargs=2, default=(100, 100))  # example max lengths
    parser.add_argument('--k_mer', type=int, default=1)
    parser.add_argument('--vocab_path', type=str, default="/home/shenweiwei/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt")
    parser.add_argument('--dataloader_num_workers', type=int, default=16)

    # 训练参数
    parser.add_argument('--device', type=str, default='cpu', choices=['gpu', 'cpu'])
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default="./output_ft/rr_inter", help='Logging directory.')

    args = parser.parse_args()

    # 设置随机种子
    paddle.seed(args.seed)

    # 设置设备
    paddle.set_device(args.device)

    # 构建分词器和模型
    tokenizer_class = NUCTokenizer  # ensure NUCTokenizer is defined
    tokenizer = tokenizer_class(k_mer=args.k_mer, vocab_file=args.vocab_path)

    # 加载预训练模型
    pretrained_model = None
    hidden_states_size = 768
    if args.with_pretrain:
        pretrained_model = ErnieModel.from_pretrained(args.model_name_or_path)
        model_config = pretrained_model.get_model_config()
        hidden_states_size = model_config["hidden_size"]

    model = ErnieRRInter(extractor=pretrained_model, hidden_states_size=hidden_states_size, proj_size=args.proj_size,
                         with_pretrain=args.with_pretrain)

    if not args.train:
        model.set_state_dict(paddle.load(args.model_path))

    print("Preparing data.")
    datasets_generator = GenerateRRInterTrainTest(rr_dir=args.dataset_dir, dataset=args.dataset, seed=args.seed)
    raw_data = datasets_generator.get()

    # 初始化训练器
    rr_inter_trainer = RRInterTrainer(args, model=model, tokenizer=tokenizer)
    rr_inter_trainer.train_dataset = RRInterDataset(raw_data, tokenizer, args.max_seq_lens)

    # 执行 k 折交叉验证
    rr_inter_trainer.cross_validate(num_folds=5)
import time
import os.path as osp
from collections import defaultdict
from functools import partial
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import argparse
import paddle
import paddle.nn as nn
from paddlenlp.data import Stack
from paddle.io import Dataset

import os.path as osp
import argparse
from functools import partial

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieModel
from paddlenlp.datasets import MapDataset

from arg_utils import (
    default_logdir,
    str2bool,
    set_seed,
    print_config,
    str2list
)
from rna_rna import (
    convert_instance_to_rr,
    RRInterMetrics,
    RRDataCollator,
    RRInterCriterionWrapper,
    ErnieRRInter,
    RRInterTrainer,
    GenerateRRInterTrainTest
)
from tokenizer_nuc import NUCTokenizer
from visualizer import Visualizer


# 定义数据集
DATASETS = ["combined_samples"]
MAX_SEQ_LEN = {
    "combined_samples": [26, 40]
}

# ========== Configuration
logger.info("Loading configuration.")
parser = argparse.ArgumentParser('Implementation of rna_rna.')

# model args
parser.add_argument('--model_name_or_path', type=str,
                    default="./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final/",
                    help='The pretrain model for feature extraction.')
parser.add_argument('--with_pretrain', type=lambda x: (str(x).lower() == 'true'),
                    default=True, help='Whether use original channels.')
parser.add_argument('--proj_size', type=int, default=64,
                    help='Project pretrained features to this size.')
parser.add_argument('--model_path',
                    type=str,
                    default="./output_ft/rr_inter/MirTarRAW/BERT,ERNIE,MOTIF,PROMPT/model_state.pdparams",
                    help='The build-in pretrained LM or the path to local model parameters.')

# data args
parser.add_argument('--dataset', type=str, default="MirTarRAW",
                    choices=DATASETS, help='The file list to train.')
parser.add_argument('--dataset_dir', type=str,
                    default="./data/ft/rr/", help='Local path for dataset.')
parser.add_argument('--k_mer', type=int, default=1,
                    help='Number of continuous nucleic acids to form a token.')
parser.add_argument('--vocab_path', type=str,
                    default="./data/vocab/", help='Local path for vocab file.')
parser.add_argument('--dataloader_num_workers', type=int,
                    default=16, help='The number of threads used by dataloader.')
parser.add_argument('--dataloader_drop_last', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Whether drop the last sample.')

# training args
parser.add_argument('--device', type=str, default='cpu', choices=['gpu', 'cpu'],  # Default to cpu
                    help='Device for training.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--disable_tqdm', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Disable tqdm display if true.')
parser.add_argument('--fix_pretrain', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Whether fix parameters of pretrained model.')

parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Whether train the model.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=100,
                    help='The number of epoch for training.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='The learning rate of optimizer.')
parser.add_argument('--metrics', type=lambda x: x.split(','),
                    default="Accuracy,Recall,Precision,F1s,AUC",
                    help='Use which metrics to evaluate model, could be concatenate by ","'
                         'and the first one will show on pbar.')

# logging args
parser.add_argument('--logging_steps', type=int, default=100,
                    help='Print logs every logging_step steps.')
parser.add_argument('--output', type=str,
                    default="./output_ft/rr_inter", help='Logging directory.')
parser.add_argument('--visualdl_dir', type=str,
                    default="visualdl", help='Visualdl logging directory.')
parser.add_argument('--save_max', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Save model with max metric.')

args = parser.parse_args()

if __name__ == "__main__":
    # ========== post process
    if ".txt" not in args.vocab_path:
        # expected: "./data/vocab/vocab_1MER.txt"
        args.vocab_path = osp.join(args.vocab_path, "vocab_" + str(args.k_mer) + "MER.txt")
    if args.model_path.split(".")[-1] != "pdparams":
        args.model_path = osp.join(args.model_path, "model_state.pdparams")

    ct = time.strftime("%Y%m%d-%H%M%S")
    args.output = osp.join(osp.join(args.output, args.dataset), ct)
    args.visualdl_dir = osp.join(args.output, args.visualdl_dir)

    # ========== Set random seeds
    logger.info("Set random seeds.")
    set_seed(args.seed)

    # ========== Set device
    logger.info("Set device.")
    paddle.set_device(args.device)

    # ========== Build tokenizer, pretrained model, model, criterion
    logger.info("Build tokenizer, pretrained model, model, criterion.")
    # load tokenizer
    logger.info("Loading tokenization.")
    tokenizer_class = NUCTokenizer
    tokenizer = tokenizer_class(k_mer=args.k_mer, vocab_file=args.vocab_path)

    # load pretrained model
    logger.info("Loading pretrained model.")
    pretrained_model = None
    hidden_states_size = 768
    if args.with_pretrain:
        pretrained_model = ErnieModel.from_pretrained(args.model_name_or_path)
        model_config = pretrained_model.get_model_config()
        hidden_states_size = model_config["hidden_size"]

    model = ErnieRRInter(extractor=pretrained_model,
                         hidden_states_size=hidden_states_size,
                         proj_size=args.proj_size,
                         with_pretrain=args.with_pretrain,
                         fix_pretrain=args.fix_pretrain)

    if not args.train:
        model.set_state_dict(paddle.load(args.model_path))

    # load criterion
    _loss_fn = RRInterCriterionWrapper()

    # ========== Prepare data
    logger.info("Preparing data.")
    datasets_generator = GenerateRRInterTrainTest(rr_dir=args.dataset_dir, dataset=args.dataset, split=1.0,
                                                  seed=args.seed)
    raw_dataset = datasets_generator.get()[0]  # 这里获取整个数据集，不分割

    # 创建 KFold 实例
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    all_metrics = []  # 用于存储所有折的结果

    for fold_idx, (train_index, val_index) in enumerate(kf.split(raw_dataset)):
        logger.info(f"Fold {fold_idx + 1}/5")

        # 根据 KFold 划分数据
        train_data = [raw_dataset.data[i] for i in train_index]
        val_data = [raw_dataset.data[i] for i in val_index]

        # 将训练和验证数据转换为 MapDataset
        m_dataset_train = MapDataset(train_data)
        m_dataset_val = MapDataset(val_data)

        # 应用映射函数
        trans_func = partial(convert_instance_to_rr, tokenizer=tokenizer, max_seq_lens=MAX_SEQ_LEN[args.dataset])
        m_dataset_train.map(trans_func)
        m_dataset_val.map(trans_func)

        # ========== Create the learning rate scheduler and optimizer
        logger.info("Creating learning rate scheduler and optimizer.")
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr)

        # ========== Create visualizer
        if args.train:
            _visualizer = Visualizer(log_dir=args.visualdl_dir,
                                     name="RNA RNA interaction, " + args.dataset + ", Fold " + str(fold_idx + 1))
        else:
            _visualizer = None

        # ========== Training
        logger.info("Start training.")
        _collate_fn = RRDataCollator(max_seq_len=sum(MAX_SEQ_LEN[args.dataset]))
        _metric = RRInterMetrics(metrics=args.metrics)

        # 重新实例化训练器
        rr_inter_trainer = RRInterTrainer(
            args=args,
            tokenizer=tokenizer,
            model=model,
            train_dataset=m_dataset_train,
            eval_dataset=m_dataset_val,
            data_collator=_collate_fn,
            loss_fn=_loss_fn,
            optimizer=optimizer,
            compute_metrics=_metric,
            visual_writer=_visualizer
        )

        if args.train:
            for i_epoch in range(args.num_train_epochs):
                logger.info(f"Epoch: {i_epoch + 1}")
                rr_inter_trainer.train(i_epoch)
                metrics = rr_inter_trainer.eval(i_epoch)
                all_metrics.append(metrics)  # 收集每轮的性能指标

    # 计算所有折的平均性能指标
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    logger.info("Cross-validation results: {}".format(avg_metrics))

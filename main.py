import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn import preprocessing
from tqdm import tqdm

from dataframe_manager import date_features
from dataframe_manager import demand_features
from dataframe_manager import sales_features
from dataloader import DataLoading
from model import LSTM
from utils import evaluate_model
from utils import read_data
from utils import train_model


def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument(
        "--state_id", default="CA_1", type=str, help="어떤 state를 선택할 것인가?",
    )
    p.add_argument(
        "--batch_size", default=512, type=int, help="batch_size",
    )
    p.add_argument(
        "--device", default="cuda", type=str, help="gpu_device",
    )
    p.add_argument(
        "--n_epochs", default=10, type=int, help="전체 epochs 수",
    )
    p.add_argument(
        "--boosting_type", default='gbdt', type=str,
    )
    p.add_argument(
        "--objective", default='tweedie', type=str,
    )
    p.add_argument(
        "--tweedie_variance_poser", default=1.1, type=float,
    )
    p.add_argument(
        "--metric", default='rmse', type=str,
    )
    p.add_argument(
        "--subsample", default=0.5, type=float,
    )
    p.add_argument(
        "--subsample_freq", default=1, type=float,
    )
    p.add_argument(
        "--leaning_rate", default=3e-3, type=float, help="learning rate",
    )
    p.add_argument(
        "--num_leaves", default=2 ** 11 - 1, type=int
    )
    p.add_argument(
        "--min_data_in_leaf", default=2 ** 12 - 1, type=int,
    )
    p.add_argument(
        "--feature_fraction", default=0.6, type=float,
    )
    p.add_argument(
        "--max_bin", default=100, type=int,
    )
    p.add_argument(
        "--n_estimators", default=1400, type=int,
    )
    p.add_argument(
        "--boost_from_average", default=False, type=bool,
    )
    p.add_argument(
        "--verbose", default=-1, type=int,
    )
    p.add_argument(
        "--target", default='sales', type=str,
    )
    p.add_argument(
        "--start_train", default=0, type=int,
    )
    p.add_argument(
        "--end_train", default=1913, type=int,
    )
    p.add_argument(
        "--p_horizon", default=28, type=int,
    )
    p.add_argument(
        "--original", default='./data/m5-forecasting-accuracy', type=str,
    )
    p.add_argument(
        "--base", default='./grid/grid_part_1.pkl', type=str,
    )
    p.add_argument(
        "--price", default='./grid/grid_part_2.pkl', type=str,
    )
    p.add_argument(
        "--calendar", default='./grid/grid_part_3.pkl', type=str,
    )
    p.add_argument(
        "--model_fn", default='./weights/', type=str,
    )
    p.add_argument(
        "--wandb", action='store_true'
    )

    config = p.parse_args()

    return config


def criterion1(pred1, targets):
    l1 = nn.MSELoss()(pred1, targets)
    return l1


if __name__ == "__main__":

    config = get_config()
    if config.wandb:
        wandb.init()
        wandb.config.update(config)

    
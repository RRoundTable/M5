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
        "--state", default="CA_1", type=str, help="어떤 state를 선택할 것인가?",
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
        "--lr", default=1e-4, type=float, help="learning rate",
    )

    p.add_argument(
        "--wandb", default=False, type=bool, help="Use wandb option",
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

    calendar, sell_prices, sales_train_validation, submission = read_data(
        "./data/m5-forecasting-accuracy",
    )
    sales_train_validation_melt = pd.melt(
        sales_train_validation,
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="day",
        value_name="demand",
    )

    sales_CA_1 = sales_train_validation_melt[
        sales_train_validation_melt.store_id == config.state
    ]
    new_CA_1 = pd.merge(sales_CA_1, calendar, left_on="day", right_on="d", how="left")
    new_CA_1 = pd.merge(
        new_CA_1,
        sell_prices,
        left_on=["store_id", "item_id", "wm_yr_wk"],
        right_on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    new_CA_1["day_int"] = new_CA_1.day.apply(lambda x: int(x.split("_")[-1]))

    CA1 = new_CA_1
    CA1 = CA1[["item_id", "day_int", "demand", "sell_price", "date"]]
    CA1.fillna(0, inplace=True)

    data_info = CA1[["item_id", "day_int"]]

    # total number of days -> 1913
    # for training we are taking data between 1800 < train <- 1913-28-28 = 1857

    train_df = data_info[(1800 < data_info.day_int) & (data_info.day_int < 1857)]

    # valid data is given last day -> 1885 we need to predict next 28days

    valid_df = data_info[data_info.day_int == 1885]

    if not os.path.exists("./something_spl"):
        os.mkdir("./something_spl")
        # Saving each item with there item name.npy
        for item in tqdm(np.unique(CA1.item_id)):
            one_item = CA1[CA1.item_id == item][["demand", "sell_price", "date"]]
            item_df = date_features(one_item)
            item_df = sales_features(item_df)
            item_df = demand_features(item_df)
            joblib.dump(item_df.values, f"something_spl/{item}.npy")

    label = preprocessing.LabelEncoder()
    label.fit(train_df.item_id)
    label.transform(["FOODS_3_827"])

    print(label)

    datac = DataLoading(train_df, label)
    n = datac.__getitem__(100)
    print(n["features"].shape, n["label"].shape)
    model = LSTM()
    model.to(config.device)

    train_dataset = DataLoading(train_df, label)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valid_dataset = DataLoading(valid_df, label)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, mode="min", factor=0.7, verbose=True, min_lr=1e-5,
    )

    for epoch in range(1, config.n_epochs + 1):
        train_loss = train_model(
            model,
            train_loader,
            criterion1,
            optimizer,
            epoch,
            scheduler=scheduler,
            history=None,
        )

        if config.wandb:
            wandb.log({"train_loss": train_loss})

        val_loss = evaluate_model(
            model, valid_loader, criterion1, epoch, scheduler=scheduler, history=None,
        )

        if config.wandb:
            wandb.log({"val_loss": val_loss})

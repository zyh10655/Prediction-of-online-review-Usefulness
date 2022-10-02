import numpy as np
import pandas as pd
import torch
import torch.utils.data as dt
import pytorch_lightning as pl
import os
import sys
import gc

sys.path.append("..")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy

from scripts.distilbert_reg import DistilBERTRegressor
from scripts.data_module import YelpDataModule, YelpPredictDataModule

from best_config import CONFIG

pl.seed_everything(seed=42)


def merge_data(df1:pd.DataFrame, df2:pd.DataFrame, on:str, suffixes:tuple=None) -> pd.DataFrame:
    """ Function to merge the dataframe """
  
    if suffixes is None:
        suffixes = ('_x', '_y')
    df_merge = pd.merge(df1, df2, on=on, suffixes=suffixes)
    df_merge = df_merge[['r_text', 'r_useful']]

    return df_merge


def train(train:pd.DataFrame):
    
    config = deepcopy(CONFIG)

    dm = YelpDataModule(config, train)

    model = DistilBERTRegressor(config)

    trainer = pl.Trainer(
        devices=-1,
        accelerator="gpu",
        precision=16,
        max_epochs=config['n_epochs'],
        gradient_clip_val=config['clip_val'],
        deterministic=True,
        strategy="ddp"
    )
    
    dm.setup()
    train_dl = dm.train_dataloader()
    trainer.fit(model, train_dl)


    trainer.save_checkpoint("./models/dbert.ckpt")


def predict_data(data:pd.DataFrame):
    dm = YelpPredictDataModule(data, CONFIG)
    model = DistilBERTRegressor(CONFIG)
    dm.setup()
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
    )

    pred = trainer.predict(model=model, dataloaders=dm.predict_dataloader(), ckpt_path="./models/dbert.ckpt")
    pred = torch.cat(pred).flatten().numpy()

    return pred


def calc_metrics(y_true, y_hat):
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mae = mean_absolute_error(y_true=y_true, y_pred=y_hat)

    return rmse, mae


if __name__ == "__main__":

    if not os.path.exists("./models/"):
        os.mkdir('./models')

    df_train_main = pd.read_parquet(CONFIG["file_paths"]["main"])
    df_train_text = pd.read_parquet(CONFIG["file_paths"]["text"])

    df_test_main = pd.read_parquet(CONFIG["test_file_paths"]["main"])
    df_test_text = pd.read_parquet(CONFIG["test_file_paths"]["text"])

    df_train = merge_data(df_train_text, df_train_main, "r_id", suffixes=("_text", "_main"))
    df_test = merge_data(df_test_text, df_test_main, "r_id", suffixes=("_text", "_main"))

    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.isfile("./models/dbert.ckpt"):
        train(df_train)

    else:
        train_preds = predict_data(df_train)
        train_targets = df_train['r_useful'].values
        rmse, mae = calc_metrics(train_targets, train_preds)

        print(f"Train RMSE: {rmse:.5f} || MAE: {mae:.5f}")

        test_preds = predict_data(df_test)
        test_targets = df_test['r_useful'].values
        rmse, mae = calc_metrics(test_targets, test_preds)

        print(f"Test RMSE: {rmse:.5f} || MAE: {mae:.5f}")










    




        
    













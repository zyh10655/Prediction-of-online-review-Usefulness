import numpy as np
import pandas as pd
import torch
import torch.utils.data as dt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# import wandb
import argparse
import sys
import os

sys.path.append("..")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from scripts.distilbert_reg import DistilBERTRegressor
from scripts.data_module import YelpDataModule, YelpPredictDataModule

from config import CONFIG

pl.seed_everything(seed=42)
# wandb.login()


def merge_data(df1:pd.DataFrame, df2:pd.DataFrame, on:str, suffixes:tuple=None) -> pd.DataFrame:
    """ Function to merge the dataframe """
  
    if suffixes is None:
        suffixes = ('_x', '_y')
    df_merge = pd.merge(df1, df2, on=on, suffixes=suffixes)
    df_merge = df_merge[['r_text', 'r_useful']]

    return df_merge
    

def predict(data, ckpt_path):
    dl = YelpPredictDataModule(data, CONFIG)
    model = DistilBERTRegressor(CONFIG)
    dl.setup()
    trainer = pl.Trainer(
        devices=-1, 
        accelerator="gpu", 
        precision=16, # Activate fp16 Training
        # strategy="ddp_find_unused_parameters_false" # For Multi-GPU Training
    )

    pred = trainer.predict(model=model, dataloaders=dl.predict_dataloader(), ckpt_path=ckpt_path)
    pred = torch.stack(pred).flatten().numpy()
    return pred


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mae

def cross_val(df:pd.DataFrame) -> None:
    """ Function to cross-validate the model """

    kfold = KFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=42)
    
    errs = defaultdict(lambda: list())

    for k, (train_idx, val_idx) in enumerate(kfold.split(df)):
        
        print(f"<---- Fold - {k+1} ----->")
        
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        # Setting up checkpoints 
        model_checkpoints = ModelCheckpoint(
            dirpath=CONFIG["checkpoint_dir_path"],
            filename=f"Fold-{k}_dbert_base",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        # Setup Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            verbose=True
        )

        # Initialize the data module
        dl = YelpDataModule(CONFIG, train, val)

        # Setup the model
        model = DistilBERTRegressor(CONFIG) 

        # Define trainer
        trainer = pl.Trainer(
            devices=-1, 
            accelerator="gpu", 
            precision=16, # Activate fp16 Training
            max_epochs = CONFIG["n_epochs"],
            callbacks=[model_checkpoints, early_stopping],
            gradient_clip_val=CONFIG["clip_val"],
            strategy="ddp_find_unused_parameters_false" # For Multi-GPU Training
            )

        trainer.fit(model, datamodule=dl)

    #     train_preds = predict(train, trainer.checkpoint_callback.best_model_path)
    #     rmse, mae = calc_metrics(train['r_useful'].values, train_preds)
    
    #     errs['train_rmse'].append(rmse)
    #     errs['train_mae'].append(mae)


    #     val_preds = predict(val, trainer.checkpoint_callback.best_model_path)
    #     rmse, mae = calc_metrics(val['r_useful'].values, val_preds)

    #     errs['val_rmse'].append(rmse)
    #     errs['val_mae'].append(mae)

    #     print(f"Train RMSE : {errs['train_rmse'][k]:.5f}, Val RMSE : {errs['val_rmse'][k]}")

    # print("*"*20)
    # print(f'Mean Train RMSE over all CV-Folds: {np.mean(errs["train_rmse"])}')
    # print(f'Mean Train MAE over all CV-Folds: {np.mean(errs["train_mae"])}')

    # print(f'Mean Validation RMSE over all CV-Folds: {np.mean(errs["val_rmse"])}')
    # print(f'Mean Validation MAE over all CV-Folds: {np.mean(errs["val_mae"])}')
    # print("*"*20)


def train_test(train, test):

    # Setting up checkpoints 
    model_checkpoints = ModelCheckpoint(
        dirpath=CONFIG["checkpoint_dir_path"],
        filename=f"dbert_reg_base",
        monitor="test_loss",
        mode="min"
    )

    early_stopping = EarlyStopping(
        monitor="test_loss",
        mode="min",
        patience=2,
        verbose=True
    )
    # Initialize the data module
    dm = YelpDataModule(CONFIG, train, test=test)
    dm.setup()
    train_dl = dm.train_dataloader()
    test_dl = dm.test_dataloader()


    # Setup the model
    model = DistilBERTRegressor(CONFIG) 

    # Define trainer
    trainer = pl.Trainer(
        devices=-1, 
        accelerator="gpu", 
        precision=16, # Activate fp16 Training
        max_epochs = CONFIG["n_epochs"],
        callbacks=[model_checkpoints, early_stopping],
        gradient_clip_val=CONFIG["clip_val"],
        
        # strategy="ddp_find_unused_parameters_false" # For Multi-GPU Training
        )

    trainer.fit(model, train_dataloaders=train_dl)
    
    # train_preds = predict(train, trainer.checkpoint_callback.best_model_path)
    # test_preds = predict(test, trainer.checkpoint_callback.best_model_path)
    # print(test_preds.shape)
    # print(test['r_useful'].values.shape)

    # rmse, mae = calc_metrics(train['r_useful'].values, train_preds)
    # print(f"Train RMSE: {rmse:.5f}, MAE: {mae:.5f}")

    # rmse, mae = calc_metrics(test['r_useful'].values, test_preds)
    # print(f"Val RMSE: {rmse:.5f}, MAE: {mae:.5f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--stage', help="whether to perform cv or simple train test", type=str)
    args = parser.parse_args()

    del CONFIG['wandb']
    del CONFIG['_wandb_kernel']

    CONFIG['batch_size']['train'] = 2
    CONFIG['batch_size']['valid'] = 4
    CONFIG['batch_size']['predict'] = 1
    
    df_train_main = pd.read_parquet(CONFIG["file_paths"]["main"])
    df_train_text = pd.read_parquet(CONFIG["file_paths"]["text"])


    df_test_main = pd.read_parquet(CONFIG["file_paths"]["main"])
    df_test_text = pd.read_parquet(CONFIG["file_paths"]["text"])


    df_train = merge_data(df_train_text, df_train_main, "r_id", suffixes=("_text", "_main"))
    df_test = merge_data(df_test_text, df_test_main, "r_id", suffixes=("_text", "_main"))


    df_train = df_train.sample(frac=0.0001)
    df_test = df_test.sample(frac=0.001)

    
    if args.stage == 'cv':
        cross_val(df_train)

    elif args.stage == 'train-test':
        CONFIG['n_epochs'] = 2
        train_test(df_train, df_test)

    # print(predict(df_train, trainer.checkpoint_callback.best_model_path).size(), df_train.shape)
    

















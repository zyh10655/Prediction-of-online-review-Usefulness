from ctypes import Union
import numpy as np
import pandas as pd
import torch
import torch.utils.data as dt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import optuna
import sys
import os
import gc
import argparse

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


def merge_data(df1:pd.DataFrame, df2:pd.DataFrame, on:str, suffixes:tuple=None) -> pd.DataFrame:
    """ Function to merge the dataframe """
  
    if suffixes is None:
        suffixes = ('_x', '_y')
    df_merge = pd.merge(df1, df2, on=on, suffixes=suffixes)
    df_merge = df_merge[['r_text', 'r_useful']]

    return df_merge



def objective(trial:optuna.Trial, train:pd.DataFrame, val:pd.DataFrame):
    
    max_len = trial.suggest_int("max_len", 128, 512, step=128)
    lr = trial.suggest_float("lr", 1e-5, 1e-4)
    t_max = trial.suggest_int("T_max", 500, 1000, step=100)
    clip_val = trial.suggest_float("clip_val", 0, 0.5, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 0, 1e-6)

    config = deepcopy(CONFIG)

    del config['wandb']
    del config['_wandb_kernel']
    
    config['max_len'] = max_len
    config['lr'] = lr
    config['scheduler']['T_max'] = t_max
    config['clip_val'] = clip_val
    config['weight_decay'] = weight_decay

    model_checkpoint = ModelCheckpoint(
        dirpath=config['checkpoint_dir_path'],
        filename=f"ckpt_distilbert_trial_{trial.number}",
        monitor="val_loss",
        mode="min"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        verbose=True
    )

    dm = YelpDataModule(config, train, val=val)

    model = DistilBERTRegressor(config)

    trainer = pl.Trainer(
        devices=-1,
        accelerator="gpu",
        precision=16,
        max_epochs=config['n_epochs'],
        callbacks=[
            early_stopping
        ],
        gradient_clip_val=config['clip_val'],
        deterministic=True,
        # detect_anomaly=True,
        # strategy="ddp_find_unused_parameters_false"
    )

    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    trainer.fit(model, train_dl, val_dl)
 
    score = trainer.test(model, val_dl)
    print("<=== SCORE===>")
    print(score)
    return score[0]['test_loss']





def objective_cv(trial:optuna.Trial) -> np.array:
    kfold = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    scores = list()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df_train)):
        torch.cuda.empty_cache()
        gc.collect()

        print(f"<====Fold - {fold}====>")
        train, val = df_train.iloc[train_idx], df_train.iloc[val_idx]
        val_loss = objective(trial, train, val)
        scores.append(val_loss)
    
    return np.mean(scores)




if __name__ == "__main__":

    df_train_main = pd.read_parquet(CONFIG["file_paths"]["main"])
    df_train_text = pd.read_parquet(CONFIG["file_paths"]["text"])

    df_train = merge_data(df_train_text, df_train_main, "r_id", suffixes=("_text", "_main"))
    df_train = df_train.sample(frac=0.0001)

    CONFIG['batch_size']['train'] = 2
    CONFIG['batch_size']['valid'] = 4


    torch.cuda.empty_cache()
    gc.collect()
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)

    study = optuna.create_study(
        study_name="distilbert-fine-tuning",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
        storage="sqlite:///study.db")

    study.optimize(
        objective_cv,
        n_trials=12,
        callbacks= [lambda study, trial: gc.collect()],
    )

    print(f"Number of finished trials : {len(study.trials)}")
    print("<==== Best Trial ====>")
    best_trial = study.best_trial
    
    print(f"Value: {best_trial.value}")
    print("<===== Params =======>")
    for key, value in best_trial.params.items():
        print(f"{key}:{value}")


    




        
    













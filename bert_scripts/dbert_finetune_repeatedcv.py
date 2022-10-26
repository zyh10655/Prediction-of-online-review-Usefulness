from typing import Tuple
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
import json

sys.path.append("..")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import partial
from copy import deepcopy
from collections import defaultdict

from scripts.distilbert_reg import DistilBERTRegressor
from scripts.data_module import YelpDataModule

from config import CONFIG

pl.seed_everything(seed=42)



def early_stopping_check(study:optuna.Study, trial:optuna.Trial, early_stopping_rounds:int=2):
    """ Function to implement the early stopping for optuna trials """

    current_trial_number = trial.number
    best_trial_number = study.best_trial.number
    should_stop = (current_trial_number - best_trial_number) >= early_stopping_rounds
    if should_stop:
        print("Early stopping detected!!")
        study.stop()


def score_model(train:pd.DataFrame, test:pd.DataFrame, hyp:dict) -> dict:

    """ Function to calculate the generalized test error of the best model """

    config = deepcopy(CONFIG)

    config['max_len'] = hyp['max_len']
    config['lr'] = hyp['lr']
    config['scheduler']['T_max'] = hyp['T_max']
    config['clip_val'] = hyp['clip_val']
    config['weight_decay'] = hyp['weight_decay']

    dm = YelpDataModule(config, train, test=test)
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()


    model = DistilBERTRegressor(config)

    trainer = pl.Trainer(
        devices=-1,
        accelerator="gpu",
        precision=16,
        max_epochs=15,
        gradient_clip_val=config['clip_val'],
        deterministic=True,
        strategy="ddp"
    )

    trainer.fit(model, train_loader)
    train_score = trainer.test(model, train_loader)
    val_score = trainer.test(model, test_loader)

    print(val_score)

    del model
    del trainer
    del train_loader
    del test_loader
    del dm

    gc.collect()
    torch.cuda.empty_cache()

    scores = {
        'train_rmse' : np.mean([train_score[i]['test_loss'] for i in range(len(train_score))]),
        'train_mae' : np.mean([train_score[i]['test_mae'] for i in range(len(train_score))]),
        'val_rmse' : np.mean([val_score[i]['test_loss'] for i in range(len(val_score))]),
        'val_mae' : np.mean([val_score[i]['test_mae'] for i in range(len(val_score))])
    }
    
    return scores




def cv_score(train:pd.DataFrame, val:pd.DataFrame, config:dict) -> float:

    """ Function to calculate the cross-validation score """

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
        strategy="ddp"
    )
    
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    score = trainer.test(model, val_dl)
    avg_score = np.mean([score[i]['test_loss'] for i in range(len(score))])

    del trainer
    del model
    del train_dl
    del val_dl
    del dm
    del score

    return avg_score



def objective(trial:optuna.Trial, data:pd.DataFrame) -> np.array:
    """ Function to select hyperparameter using cross-validation """


    kfold = KFold(n_splits=CONFIG['repeated']['folds'], shuffle=True, random_state=42)
    scores = list()


    # Hyperparameters for the model 
    max_len = trial.suggest_int("max_len", 128, 512, step=128)
    lr = trial.suggest_float("lr", 1e-6, 1e-4)
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

    print(f'Current Trial:{trial.number}')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):

        print(f"<==== Fold - {fold}====>")
        train, val = data.iloc[train_idx], data.iloc[val_idx]
        val_loss = cv_score(train, val, config)
        scores.append(val_loss)

        torch.cuda.empty_cache()
        gc.collect()

    
    return np.mean(scores)



def repeated_cross_val(train, val):
    best_hyps = list()
    scores_final = defaultdict(lambda: list())

    for rep in range(CONFIG["repeated"]["rep"]):
        print(f"Repetition : {rep + 1} ")

        # train, val = data.iloc[train_idx], data.iloc[val_idx]
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=1234)
        study = optuna.create_study(
            study_name = f"dbert-fine-tuning-rep-{rep+1}",
            direction = "minimize",
            sampler = sampler, 
            # pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2),
            load_if_exists=True,
            storage=f"sqlite:///study.db"
        )
        train = train.sample(frac=1, random_state=rep)
        func = lambda trial: objective(trial, train)

        study.optimize(
            func,
            n_trials=50,
            callbacks=[partial(early_stopping_check, early_stopping_rounds=2)]
        )

        print(f"Number of finished trial : {len(study.trials)}")
        print(f"Best Trial Value : {study.best_trial.value}")

        best_hyp = dict()
        for k, v in study.best_trial.params.items():
            print(f"{k}:{v}")
            best_hyp[k] = v

        best_hyps.append(best_hyp)


        scores = score_model(train, val, best_hyp)
        print(f'Training Score (Outer fold) RMSE - {scores["train_rmse"]:.5f}')
        print(f'Training Score (Outer fold) MAE - {scores["train_mae"]:.5f}')

        print(f'Validation Score (Outer fold) RMSE - {scores["val_rmse"]:.5f}')
        print(f'Validation Score (Outer fold) MAE - {scores["val_mae"]:.5f}')

        scores_final['train_rmse'].append(scores["train_rmse"])
        scores_final['train_mae'].append(scores["train_mae"])
        
        scores_final['val_rmse'].append(scores["val_rmse"])
        scores_final['val_mae'].append(scores["val_mae"])

        torch.cuda.empty_cache()
        gc.collect()

    print(f"Generalized TRAINING RMSE : {np.mean(scores_final['train_rmse'])}")
    print(f"Generalized TRAINING MAE : {np.mean(scores_final['train_mae'])}")
    print(f"Generalized TRAINING RMSE STD : {np.std(scores_final['train_rmse'])}")
    print(f"Generalized TRAINING MAE STD : {np.std(scores_final['train_mae'])}")

    print(f"Generalized VALIDATION RMSE : {np.mean(scores_final['val_rmse'])}")
    print(f"Generalized VALIDATION MAE: {np.mean(scores_final['val_mae'])}")
    print(f"Generalized TRAINING RMSE STD : {np.std(scores_final['val_rmse'])}")
    print(f"Generalized TRAINING MAE STD : {np.std(scores_final['val_mae'])}")

    with open("best_hyps.json", "w") as fp:
        json.dump(best_hyps, fp)

    return scores_final




if __name__ == "__main__":

    df_train = pd.read_parquet(CONFIG["file_paths"]["train"])
    df_val = pd.read_parquet(CONFIG["file_paths"]["val"])
    # scores = nested_cross_val(df_train)
    scores = repeated_cross_val(df_train, df_val)
    print(scores)

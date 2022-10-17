import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertModel

CONFIG = {
    "n_epochs" : 50,
    "n_folds" : 3,
    "nested":{
        "inner":3,
        "outer":3
    },
    "repeated":{
        "rep":6,
        "folds": 5
    },
    "lr" : 1e-4,
    "batch_size":{
        "train":64,
        "valid": 64,
        "test":128,
        "predict":128,
    },
    'persistent_workers':True,
    "max_len":300,
    "bert":{
        "name":"distilbert-base-uncased",
        "tokenizer": DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'),
        "config" : DistilBertConfig.from_pretrained("distilbert-base-uncased"),
        'out_attention':False
    },
    "fc":{
        "linear1": 768,
        "linear2": 128
    },
    "dropout":0.25,
    "criterion":nn.MSELoss(),
    "file_paths":{
        "train":"../data/subsampled/100K35F_train_text.parquet.snappy",
        "val" : "../data/subsampled/100K35F_val_text.parquet.snappy",
        "test":"../data/subsampled/100K35F_test_text.parquet.snappy"
    },
    "num_workers":4,
    "pin_memory":True,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "_wandb_kernel":"neuracort",
    "wandb":True,
    "scheduler":{
        "name":"CosineAnnealingLR",
        # "name":"",
        "min_lr":1e-6,
        "T_0":50,
        "T_max":500
    },
    "checkpoint_dir_path":"./checkpoints/",
    "weight_decay":1e-6,
    "clip_val":2
}

import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertConfig

CONFIG = {
    "n_epochs" : 10,
    "n_folds" : 3,
    "lr" : 3.48e-5,
    "batch_size":{
        "train":64,
        "valid": 128,
        "test":128,
        "predict":128,
    },
    'persistent_workers':True,
    "max_len":300,
    "bert":{
        "name":"distilbert-base-uncased",
        "tokenizer": DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'),
        "config" : DistilBertConfig.from_pretrained("distilbert-base-uncased"),
        'out_attention':True
    },
    "fc":{
        "linear1": 768,
        "linear2": 128
    },
    "dropout":0.25,
    "criterion":nn.MSELoss(),
    "file_paths":{
        "main":"../data/new_data/train_main.parquet.snappy",
        "text":"../data/new_data/train_text.parquet.snappy"
    },
    "test_file_paths":{
        "main":"../data/new_data/test_main.parquet.snappy",
        "text":"../data/new_data/test_text.parquet.snappy"
    },
    "num_workers":4,
    "pin_memory":True,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "_wandb_kernel":"neuracort",
    "wandb":True,
    "scheduler":{
        "name":"CosineAnnealingLR",
        "min_lr":1e-6,
        "T_0":50,
        "T_max":900
    },
    "checkpoint_dir_path":"./checkpoints/",
    "weight_decay":8.75e-7,
    "clip_val":0.5
}

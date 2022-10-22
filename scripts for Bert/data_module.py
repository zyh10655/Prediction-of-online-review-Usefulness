import pandas as pd
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class YelpDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = int(self.targets[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length', 
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'target':torch.tensor(target, dtype=torch.long)
        }


class YelpDataModule(pl.LightningDataModule):
    def __init__(self,
        config:dict,
        train:pd.DataFrame,
        val:pd.DataFrame=None,
        test:pd.DataFrame=None
        ):

        super().__init__()

        self.save_hyperparameters(logger=False) # Will allow access to __init__ varaibles through `self.hparams` attribute
        
        self.transforms = None
        self.train = train
        self.val = val
        self.test = test
    
    def setup(self, stage:str=None) -> None:

        self.train = YelpDataset(
            self.train['r_text'].values, 
            self.train['r_useful'].values, 
            self.hparams.config["bert"]["tokenizer"], 
            self.hparams.config["max_len"])

        if self.val is not None:
            self.val = YelpDataset(
                self.val['r_text'].values, 
                self.val['r_useful'].values, 
                self.hparams.config["bert"]["tokenizer"], 
                self.hparams.config["max_len"])

        if self.test is not None:
            self.test = YelpDataset(
                self.test['r_text'].values,
                self.test['r_useful'].values,
                self.hparams.config["bert"]["tokenizer"],
                self.hparams.config["max_len"]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train, 
            batch_size=self.hparams.config["batch_size"]["train"], 
            num_workers=self.hparams.config["num_workers"], 
            pin_memory=self.hparams.config["pin_memory"], 
            persistent_workers=self.hparams.config['persistent_workers'],
            shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val, 
            batch_size=self.hparams.config["batch_size"]["valid"], 
            num_workers=self.hparams.config["num_workers"], 
            pin_memory=self.hparams.config["pin_memory"], 
            persistent_workers=self.hparams.config['persistent_workers'],
            shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test, 
            batch_size=self.hparams.config["batch_size"]["test"], 
            num_workers=self.hparams.config["num_workers"], 
            pin_memory=self.hparams.config["pin_memory"], 
            persistent_workers=self.hparams.config['persistent_workers'],
            shuffle=False)



class YelpPredictDataModule(pl.LightningDataModule):
    def __init__(self, data:pd.DataFrame, config:dict):
        self.config = config
        self.data = data

    def setup(self, stage:str=None) -> None:
        self.data = YelpDataset(
            self.data['r_text'].values,
            self.data['r_useful'].values,
            self.config['bert']['tokenizer'],
            self.config['max_len'])

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=self.config['batch_size']['predict'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=self.config['persistent_workers'],
            shuffle=False
        )

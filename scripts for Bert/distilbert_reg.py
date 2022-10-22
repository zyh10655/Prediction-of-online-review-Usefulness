from cmath import isnan
from os import sync
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

from transformers import get_linear_schedule_with_warmup, DistilBertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_scheduler(optimizer, config):
    if config["scheduler"]["name"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config["scheduler"]["T_max"],
            eta_min=config["scheduler"]["min_lr"],
        )
    elif config["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
        scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["scheduler"]["T_0"],
            eta_min=config["scheduler"]["min_lr"]
        )
    else:
        scheduler = None

    return scheduler


class DistilBERTRegressor(pl.LightningModule):
    def __init__(self, config):
        
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.dbert = DistilBertModel.from_pretrained(config['bert']['name'], output_attentions=config['bert']['out_attention'])

        self.head = nn.Sequential(
            nn.Linear(self.dbert.config.hidden_size, self.config['fc']['linear1']),
            nn.ReLU(),
            nn.Dropout(p=self.config['dropout']),
            nn.Linear(self.config['fc']['linear1'], 1)
        )

        self.output = nn.Linear(self.dbert.config.hidden_size, 1)
        self.metric = torchmetrics.MeanSquaredError(squared=False, compute_on_step=False)
        

    def forward(self, input_ids, attention_mask):
        dbert_out = self.dbert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict=True
        )

        last_hidden_state = dbert_out.last_hidden_state
        cls_token = last_hidden_state[:, 0, :] # Since CLS Token is at the begining of the sentence.

        # yhat = self.head(cls_token)
        yhat = self.output(cls_token)

        return yhat

    def compute_loss(self, yhat, y):
        y = y.reshape(-1, 1)
        return F.mse_loss(yhat, y.type_as(yhat))


    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)
        
        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss
        mae = F.l1_loss(outputs, targets.reshape(-1, 1).type_as(outputs))

        self.log("train_loss", torch.sqrt(loss), prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("train_mae", mae, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)

        return {'loss' : loss}

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)

        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss

        self.log("val_loss", torch.sqrt(loss), prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)

        return {
            'val_loss' : loss.item()
        }

    def test_step(self, batch, batch_idx):
        
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)

        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss
        mae = F.l1_loss(outputs, targets.reshape(-1, 1).type_as(outputs))
        self.log("test_loss", torch.sqrt(loss), prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("test_mae", mae, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)

        return {
            'test_loss':loss.item(),
            "test_mae": mae.item()
        }
    


    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)

        return outputs

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = get_scheduler(optimizer, self.config)

        if scheduler is None:
            return dict(
                optimizer=optimizer
            )

        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )


from cmath import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

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
        self.dbert = DistilBertModel.from_pretrained(config['bert']['name'])

        self.head = nn.Sequential(
            nn.Linear(self.dbert.config.hidden_size, self.config['fc']['linear1']),
            nn.ReLU(),
            nn.Dropout(p=self.config['dropout']),
            nn.Linear(self.config['fc']['linear1'], 1)
        )

        self.output = nn.Linear(self.dbert.config.hidden_size, 1)

        

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

        if torch.isnan(outputs).any():
            print(input_ids)
            print(attention_mask)
            raise Exception("NaN encountered")
        
        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss

        self.log("train_loss", torch.sqrt(loss), prog_bar=True, logger=True, sync_dist=True)

        return {
            'loss' : loss,
        }

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)

        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss

        self.log("val_loss", torch.sqrt(loss), prog_bar=True, logger=True, sync_dist=True)

        return {
            'val_loss' : loss
        }

    def test_step(self, batch, batch_idx):
        
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        outputs = self(input_ids, attention_mask)

        loss = self.compute_loss(outputs, targets.type_as(outputs)) # Calculates the loss

        return {
            'test_loss':loss
        }


    def training_step_end(self, outputs):
        # outputs = torch.stack([x["loss"] for x in outputs])
        outputs = outputs['loss']
        loss = outputs.mean()
        self.log('train_loss', torch.sqrt(loss))

    def validation_step_end(self, outputs):
        # outputs = torch.stack([x["val_loss"] for x in outputs])
        outputs = outputs['val_loss']
        loss = outputs.mean()
        self.log('val_loss', torch.sqrt(loss))

    def test_step_end(self, outputs):
        # outputs = torch.stack([x["test_loss"] for x in outputs])
        outputs = outputs['test_loss']
        loss = outputs.mean()
        self.log('test_loss', torch.sqrt(loss))

    


    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
        return self(input_ids, attention_mask)

    def on_predict_epoch_end(self, results):
        if self.trainer.is_global_zero:
            all_preds = self.all_gather(results[0])
            return all_preds
    
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


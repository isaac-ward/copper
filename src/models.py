import torch
import torch.nn
import torchvision.models

import pytorch_lightning as pl

class BaselineModel(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        
        # We'll have a feature extractor backbone going into a regression
        self.model    = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
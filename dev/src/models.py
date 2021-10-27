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

        # We'll use mean squared error
        self.loss_computer = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _get_output_key_as_tensor(self, outputs, key):
        return torch.stack([o[key] for o in outputs])

    def _log_step_scalar(self, label, scalar):
        self.log(
            label, 
            scalar,               
            on_step=True, 
            on_epoch=False, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )

    def _log_epoch_scalar(self, label, scalar):
        self.log(
            label, 
            scalar,    
            on_step=False, 
            on_epoch=True, 
            prog_bar=False, 
            logger=True, 
            sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        # Calculate and log the step loss
        x, y = batch
        y_hat = self(x)
        loss = self.loss_computer(y_hat, y)

        self._log_step_scalar('train_loss_step', loss)
        
        return {
            'loss': loss
        }

    def training_step_end(self, outputs):
        # This needs to be passed through in parallel processing mode, otherwise
        # nothing will be sent to the validation epoch end function
        return outputs

    def training_epoch_end(self, outputs):
        # Calculate and log the epoch loss
        avg_loss = self._get_output_key_as_tensor(outputs, 'loss').mean()
        self._log_epoch_scalar('train_loss_epoch', avg_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {
            'y': y, 
            'y_hat': y_hat
        }

    def validation_step_end(self, outputs):
        # This needs to be passed through in parallel processing mode, otherwise
        # nothing will be sent to the validation epoch end function
        return outputs

    def validation_epoch_end(self, outputs):
        y     = self._get_output_key_as_tensor(outputs, 'y')
        y_hat = self._get_output_key_as_tensor(outputs, 'y_hat')

        # Mean squared error
        mse = self.loss_computer(y_hat, y)
        self._log_epoch_scalar('val_mse_epoch', mse)

        # Calculate skill
        # First need standard deviation of true labels
        std = torch.std(y)
        skill = 1 - ( mse / std ) ** 0.5
        self._log_epoch_scalar('val_skill_epoch', skill)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
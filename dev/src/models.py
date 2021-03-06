import torch
import torch.nn
import torchvision.models

import pytorch_lightning as pl


# Needed to remove extra dimensions in model
class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze()

class Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        
        # Need to do this squeeze fix for every model
        self.model = torch.nn.Sequential(
            #self._setup_barlow_resnet(),
            self._setup_dino_transformer(),
            #self._setup_dino_resnet(),
            #self._setup_resnet(pretrained=False),
            #self._setup_simple_convnet(),
            Squeeze()
        )

        # We'll use mean squared error
        self.loss_computer = torch.nn.MSELoss()
    
    def _setup_dino_transformer(self):
        # TODO - why does this collapse?
        model = torch.nn.Sequential(
            # 21M parameters 
            torch.hub.load('facebookresearch/dino:main', 'dino_vits8'),
            # 0.5M parameters
            torch.nn.Linear(384, 1)
        )
        return model

    def _setup_dino_resnet(self):
        model    = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model.fc = torch.nn.Linear(2048, 1)
        return model

    def _setup_barlow_resnet(self):
        model    = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        model.fc = torch.nn.Linear(2048, 1)
        return model

    def _setup_resnet(self, pretrained):
        # 23M parameters
        model    = torchvision.models.resnet50(pretrained=pretrained)
        # 0.5M parameters
        model.fc = torch.nn.Linear(2048, 1)
        return model

    def _setup_simple_convnet(self):

        # As discussed in the paper - this is the simple convnet
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding_mode='zeros'),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4,  stride=2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4,  stride=4),

            torch.nn.Flatten(),

            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 1),
        )
        return model

    def _setup_complex_convnet(self):

        # As discussed in the paper - this is the 5L convnet (see the Conv2D_5L_Batch file in the Networks.py file in their code)
        # TODO - run this experiment! It might be the model they use to get the elusive 0.36 skill!
        #model = torch.nn.Sequential(
            # TODO
        #)
        #return model

        pass

    def forward(self, x):
        return self.model(x)

    def _get_output_key_as_tensor(self, outputs, key):
        return torch.cat([o[key] for o in outputs], dim=0)

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
        denom = 0.33 #torch.std(y) ** 2
        skill = 1 - mse ** 0.5 / denom
        self._log_epoch_scalar('val_skill_epoch', skill)

    # TODO: testing

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
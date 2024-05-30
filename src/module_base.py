### Define abstract methods for generic training (optimizer, checkpointing, etc) ###

import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

class BaseModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        save_dir: str = None,
        **kwargs,
    ):
        super().__init__()

        """ We pass in save_dir to help us with locating the model
        and (if needed) for saving intermediate outputs like TSNE
        For the lightning purists, I acknowledge that this
        really should be done with Callbacks - M
        """
        self.save_dir = save_dir

        # loss, cache and metrics are specific for each sub-module
        self.loss_fcn = None
        self.metrics = None
        self.cache = None # cache for saving intermediate inputs

    def forward(self, x):
        raise NotImplementedError("The user should implement method in submodule")

    def loss_wrapper(self, y1, y2):
        raise NotImplementedError("The user should implement method in submodule")

    # This pseudocode will not function but offers some ideas of what goes here
    def common_step(self, batch, batch_idx, mode="train"):
        # obtain data from the batch
        x = batch["x"]
        y = batch["y"] # one-hot label
        y_u = batch["y_u"] # uncertainty-augmented label
        
        y1, y2 = self.forward(x)

        # compute losses
        loss = self.loss_wrapper(y1, y2)

        # update metrics here
        ###
        
        # return the loss, as PTL handles the backward() call internally
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        # return all information pertaining to the batch + the output
        raise NotImplementedError("The user should implement method in submodule")

    def configure_optimizers(self):
        # this function is called by PTL during training loop
        # should vary based on the module
        return NotImplementedError("The user should implement method in submodule")

    # This pseudocode will not function but offers some ideas of what goes here
    def common_epoch_end(self, mode="train"):
        # compute and log the mode-related metrics (to wandb, etc)
        ###

        # reset the metrics
        ###

        # save a copy of the cached predictions
        ###
        
        # clear the cache
        ###
        return

    def on_train_epoch_end(self):
        self.common_epoch_end("train")

    def on_validation_epoch_end(self):
        self.common_epoch_end("val")

    def on_test_epoch_end(self):
        self.common_epoch_end("test")
        
    # load only weights from a checkpoint file
    ### Question: is the requires_grad state part of the state_dict?
### Inherit from module_base and specify supervised image training fcns ###

import numpy as np
import os
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

import torch
import lightning.pytorch as pl
import torchmetrics

from module_base import BaseModule
from models import get_backbone
from utils import plot_emb, save_csv

class SupervisedImageModule(BaseModule):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        learning_rate: float = 1e-4,
        save_dir: str = None,
        loss='ce',
        **kwargs,
    ):
        super().__init__(encoder, decoder, save_dir, **kwargs)
        self.save_hyperparameters()

        # create directories to save the CSV file and embedding visualizations
        self.emb_dir = os.path.join(self.save_dir, "embeddings")
        self.csv_dir = os.path.join(self.save_dir, "csvs")
        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # this loss should be usable with both confidence (N, C) and integer (N,) labels
        if loss == 'ce':
            self.loss_fcn = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f"Loss function name {loss} not found, expected 'ce' ")

        # Define metrics for each stage and save intermediate outputs
        """
        Strictly speaking, the flow to save embeddings for train epoch
        should be to train the epoch with gradients, then gradient off
        and run the images through training set again to get embeddings
        but it would take more time. Here we are a bit lazy and directly
        save the embeddings the model develop during the training epoch.
        For visualization purposes, this is faster but cause earlier epochs
        to look more chaotic
        """
        metrics = {}
        self.cache = {}
        self.modes = ["train", "val", "test"]
        for j in self.modes:
            metrics[j + "_f1"] = torchmetrics.classification.MulticlassAccuracy(
               num_classes=num_classes, average="macro"
            )
            metrics[j + "_acc"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.cache[j + "_z"] = []
            self.cache[j + "_y"] = []
            self.cache[j + "_uid"] = []
            self.cache[j + "_pred"] = []
        self.metrics = torch.nn.ModuleDict(metrics)

        # Define optimizer and scheduler
        self.lr = learning_rate

    def forward(self, x):
        z = self.encoder(x) # N, D
        logits = self.decoder(z) # N, C
        return {"logits": logits, "z": z}

    def loss_wrapper(self, logits, y):
        return self.loss_fcn(logits, y)

    def common_step(self, batch, batch_idx, mode="train"):
        x = batch["x"]
        y = batch["y"] # one-hot label
        
        outs = self.forward(x)

        # compute losses
        loss = self.loss_wrapper(outs["logits"], y)

        # update metrics
        acc = self.metrics[mode + "_acc"](outs["logits"][:, : self.num_classes], y)
        f1 = self.metrics[mode + "_f1"](outs["logits"][:, : self.num_classes], y)

        log = {mode + "_loss": loss, mode + "_acc": acc, mode + "_f1": f1}
        self.log_dict(log)
        
        # update cache
        self.cache[mode + "_z"].append(outs["z"].detach().cpu())
        self.cache[mode + "_y"].append(y.cpu())
        self.cache[mode + "_uid"].extend(batch["uid"])
        self.cache[mode + "_pred"].append(outs["logits"].detach().cpu())

        return loss

    def predict_step(self, batch, batch_idx):
        # reveal all relevant input information
        x = batch["x"]
        outs = self.forward(x)
        batch.update(outs)
        return batch  # , logits, z, attn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def plot_emb_wrapper(self, z, y, fig_path, title):
        if len(z) > 50:  # define minimum number of points for plotting
            plot_emb(
                z,
                y,
                fig_path,
                protos=None,
                y_protos=None,
                title=title,
                compression="pca",
            )

    def common_epoch_end(self, mode="train"):
        # print the mode-related metrics and log them
        acc_epoch = self.metrics[mode + "_acc"].compute()
        f1_epoch = self.metrics[mode + "_f1"].compute()
        print(f"{mode}: acc_epoch: {acc_epoch}, f1_epoch: {f1_epoch}")
        self.log_dict({mode + "_acc_epoch": acc_epoch, mode + "_f1_epoch": f1_epoch})

        # reset the metric in question
        self.metrics[mode + "_acc"].reset()
        self.metrics[mode + "_f1"].reset()

        # plot the embedding visualization
        z_saved = torch.cat(self.cache[mode + "_z"]).numpy()
        y_saved = torch.cat(self.cache[mode + "_y"]).numpy()
        title = f"{mode}_{self.current_epoch}_{f1_epoch:.2f}"
        emb_save_name = title + ".jpg"
        emb_save_path = os.path.join(self.emb_dir, emb_save_name)
        self.plot_emb_wrapper(z_saved, y_saved, emb_save_path, title)

        # save a copy of the cached predictions
        pred_saved = torch.cat(self.cache[mode + "_pred"]).numpy()
        uid_saved = self.cache[mode + "_uid"]
        csv_save_name = title + ".csv"
        csv_save_path = os.path.join(self.csv_dir, csv_save_name)
        save_csv(uid_saved, y_saved, pred_saved, csv_save_path)
        
        # plot the confusion matrix
        pred_saved_argmax = np.argmax(pred_saved, axis=1)
        print(confusion_matrix(y_saved, pred_saved_argmax))
        
        for k in self.cache.keys():
            if mode in k:
                self.cache[k].clear()
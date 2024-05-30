### Inherit from module_base and specify SSL image training fcns ###

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
        teacher_encoder: nn.Module = None,
        teacher_decoder: nn.Module = None,
        learning_rate: float = 1e-4,
        save_dir: str = None,
        **kwargs,
    ):
        super().__init__(encoder, decoder, save_dir, **kwargs)
        self.save_hyperparameters()
        
        # define the encoder and projector
        self.s_enc = encoder
        self.s_proj = decoder
        
        # the teacher modules are identical to the student
        """
        In theory, the DINO encoder and projector has the same architecture and initialization
        However, there are some issues with deepcopy and its interactions with DDP
        as such, I am instantiating the same modules twice and putting them
        into the function call -M
        """
        self.t_enc = teacher_encoder
        self.t_proj = teacher_decoder

        self.loss_fcn = DinoLoss()

        # Define optimizer and scheduler
        self.lr = learning_rate
        # TODO scheduler

    def forward(self, x):
        # TODO

    def extract_features(self, x):
        # TODO
    
    def loss_wrapper(self, logits, y):
        # TODO

    def common_step(self, batch, batch_idx, mode="train"):
        x1, x2 = batch["x"][0], batch["x"][1]
        
        s1, s2, t1, t2 = self.forward(x)

        # compute losses
        loss = self.loss_wrapper(x1, x2)

        log = {mode + "_loss": loss}
        self.log_dict(log)

        return loss

    def predict_step(self, batch, batch_idx):
        # TODO provide inference using the teacher model
        return self.extract_features(batch["x"][0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # add to this more as we track more statistics
    def common_epoch_end(self, mode="train"):
        return
        # # plot the embedding visualization
        # z_saved = torch.cat(self.cache[mode + "_z"]).numpy()
        # title = f"{mode}_{self.current_epoch}_{f1_epoch:.2f}"
        # emb_save_name = title + ".jpg"
        # emb_save_path = os.path.join(self.emb_dir, emb_save_name)
        # self.plot_emb_wrapper(z_saved, y_saved, emb_save_path, title)

        # # save a copy of the cached predictions
        # pred_saved = torch.cat(self.cache[mode + "_pred"]).numpy()
        # uid_saved = self.cache[mode + "_uid"]
        # csv_save_name = title + ".csv"
        # csv_save_path = os.path.join(self.csv_dir, csv_save_name)
        # save_csv(uid_saved, y_saved, pred_saved, csv_save_path)
        
        # # plot the confusion matrix
        # pred_saved_argmax = np.argmax(pred_saved, axis=1)
        # print(confusion_matrix(y_saved, pred_saved_argmax))
        
        # for k in self.cache.keys():
            # if mode in k:
                # self.cache[k].clear()
"""
Trainer for the deep insight decoder.
"""
# --------------------------------------------------------------

import time
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import wandb

# --------------------------------------------------------------

class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        use_wandb: bool
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = (criterion[0], {key: torch.tensor(value).to(self.device) for (key,value) in criterion[1].items()})
        self.optimizer = optimizer
        self.step = 0
        self.epoch = 0
        self.use_wandb = use_wandb

    def train(
        self,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, self.train_loader.dataset.epochs):
            print(f"Beginning Epoch {epoch}")
            if self.use_wandb: wandb.log({'step': self.step, 'epoch': epoch})
            epoch_steps = 0
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                if epoch_steps > 0 and epoch_steps % self.train_loader.dataset.steps_per_epoch==0:
                    break
                epoch_steps += 1
                #print(f"Beginning Step {self.step}")
                batch = batch.to(self.device)
                for i in range(len(labels)):
                    labels[i] = labels[i].to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                losses = torch.tensor([]).to(self.device)
                for ind, logit in enumerate(logits):
                    loss_func = list(self.criterion[0].values())[ind]
                    loss_weight= list(self.criterion[1].values())[ind]
                    loss_key = list(self.criterion[0].keys())[ind]
                    # if logit.shape[1] == 1:
                    #     labels[ind] = torch.unsqueeze(labels[ind], 1)
                        #logit = torch.squeeze(logit)
                    l = torch.multiply(
                            loss_func(logit, labels[ind]) ,
                            loss_weight
                        )
                    if self.use_wandb:
                        wandb.log({'epoch': epoch, f'Training_Loss_{loss_key}': torch.sum(l)})
                        wandb.log({'step': self.step, f'Training_Loss_{loss_key}': torch.sum(l)})
                    losses = torch.cat((
                        losses,
                        l
                    ))
                loss = torch.sum(losses)
                #print(f"Loss = {loss}")
                if self.use_wandb:
                    wandb.log({'epoch': epoch, 'Training_Loss_Total': loss})
                    wandb.log({'step': self.step, 'Training_Loss_Total': loss})

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.epoch = epoch

                self.step += 1

            self.validate()
            # if ((self.epoch + 1) % self.train_loader.dataset.validation_steps) == 0:
            #     self.validate()
            # self.validate() will put the model in validation mode,
            # so we have to switch back to train mode afterwards
            self.model.train()

    def validate(self):
        total_loss = 0
        total_loss_key = {}
        self.model.eval()
        print("Validating")
        # No need to track gradients for validation as we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                for i in range(len(labels)):
                    labels[i] = labels[i].to(self.device)
                logits = self.model(batch)
                losses = torch.tensor([]).to(self.device)
                for ind, logit in enumerate(logits):
                    logit.to(self.device)
                    loss_func = list(self.criterion[0].values())[ind]
                    loss_weight = list(self.criterion[1].values())[ind]
                    loss_key = list(self.criterion[0].keys())[ind]
                    # if logit.shape[1] == 1:
                    #     labels[ind] = torch.unsqueeze(labels[ind], 1)
                    l = torch.multiply(
                            loss_func(logit, labels[ind]),
                            loss_weight
                        )
                    if loss_key in total_loss_key:
                        total_loss_key[loss_key] += torch.sum(l)
                    else:
                        total_loss_key[loss_key] = torch.sum(l)
                    if self.use_wandb:
                        wandb.log({'step': self.step, f'Validation_Loss_{loss_key}': torch.sum(l)})
                        wandb.log({'epoch': self.step, f'Validation_Loss_{loss_key}': torch.sum(l)})
                    losses = torch.cat((
                        losses,
                        l
                    ))
                loss = torch.sum(losses)
                total_loss += loss.item()

        if self.use_wandb: wandb.log({'step': self.step, 'Validation_Loss_Total': total_loss})
        print(f"Total Loss: {total_loss}")
        for k in total_loss_key.keys():
            print(f"Total Loss {k}: {total_loss_key[k]}")


    @staticmethod
    def __combine_file_results(results):
        new_res = {"preds": [], "labels": []}
        for res in results.values():
            new_res["preds"].append(np.argmax(np.mean(res["preds"], axis=0)))
            new_res["labels"].append(np.round(np.mean(res["labels"])).astype(int))
        return new_res

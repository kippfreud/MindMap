"""
Trainer for the deep insight decoder
"""
# --------------------------------------------------------------

import time
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

# --------------------------------------------------------------

wandb.init(project="my-project")

class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            print(f"Beginning Epoch {epoch}")
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                print(f"Beginning Step {self.step}")
                batch = batch.to(self.device)
                # labels = labels.to(self.device)
                data_load_end_time = time.time()

                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                output = self.model.forward(batch)

                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`
                logits = output

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                #loss = self.criterion(logits, labels)
                losses = torch.tensor([])
                for ind, logit in enumerate(logits):
                    loss_func = list(self.criterion[0].values())[ind]
                    loss_weight= list(self.criterion[1].values())[ind]
                    losses = torch.cat((
                        losses,
                        torch.multiply(
                            loss_func(logit, labels[ind]) ,
                            torch.tensor(loss_weight)
                        )
                    ))
                loss = torch.sum(losses)
                print(f"Loss = {loss}")
                wandb.log({'step': self.step, 'tr_loss': loss})
                ## TASK 10: Compute the backward pass
                # Now we compute the backward pass, which populates the `.grad` attributes of the parameters
                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.step += 1

                if ((self.step + 1) % val_frequency) == 0:
                    self.validate()
                    # self.validate() will put the model in validation mode,
                    # so we have to switch back to train mode afterwards
                    self.model.train()

    def validate(self):
        # results = {"preds": [], "labels": []}
        results = {}
        total_loss = 0
        self.model.eval()
        print("Validating")
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                #labels = labels.to(self.device)
                logits = self.model(batch)
                losses = torch.tensor([])
                for ind, logit in enumerate(logits):
                    logit.to(self.device)
                    loss_func = list(self.criterion[0].values())[ind]
                    loss_weight = list(self.criterion[1].values())[ind]
                    losses = torch.cat((
                        losses,
                        torch.multiply(
                            loss_func(logit, labels[ind]),
                            torch.tensor(loss_weight)
                        )
                    ))
                loss = torch.sum(losses)
                total_loss += loss.item()

        wandb.log({'step': self.step, 'val_loss': total_loss})
        print(f"Total Loss: {total_loss}")


    @staticmethod
    def __combine_file_results(results):
        new_res = {"preds": [], "labels": []}
        for res in results.values():
            new_res["preds"].append(np.argmax(np.mean(res["preds"], axis=0)))
            new_res["labels"].append(np.round(np.mean(res["labels"])).astype(int))
        return new_res

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from device import device

import matplotlib.pyplot as plt
import numpy as np

class Trainer():
    def __init__(self, model : nn.Module, epochs : int, loss_func : nn.Module, 
          optimizer : nn.Module, train_set : data.DataLoader, val_set : data.DataLoader = None):
        self.model = model
        self.epochs = epochs
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_set = train_set
        self.val_set = val_set
        

    def train(self) -> nn.Module:

        self.train_losses, self.val_losses = [], []

        for e in range(self.epochs):
            self.model.train()
            loss_mean = 0
            lm_count = 0

            train_tqdm = tqdm(self.train_set, leave=False)
            for X_train, y_train in train_tqdm:

                X_train = X_train.to(device)
                y_train = y_train.to(device)

                preds = self.model(X_train)
                loss = self.loss_func(preds, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
                train_tqdm.set_description(f"Epoch [{e + 1}/{self.epochs}], loss_mean={loss_mean:.3f}")

            if self.val_set:
                self.model.eval()
                Q_val, count_val = 0, 0
                for X_val, y_val in self.val_set:

                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    with torch.no_grad():
                        val_preds = self.model(X_val)
                        loss = self.loss_func(val_preds, y_val).item()
                        Q_val += loss
                        count_val += 1
                Q_val /= count_val
                self.val_losses.append(Q_val)
            
            self.train_losses.append(loss_mean)
            print(f' | loss_mean={loss_mean:.3f}', end='')
            print(f', Q_val={Q_val:.3f}' if self.val_set else '')

            st = self.model.state_dict()
            torch.save(st, f'src/CNN/models/CNN_{e + 1}.tar')
        
        return self.model
    
    def train_plot(self, title : str = None, save = False):
        plt.figure(figsize=(10, 6))

        plt.plot(self.train_losses, label='Train losses')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation losses')

        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.xticks(np.arange(0, len(self.train_losses), dtype=np.int16))

        plt.title(label=title, fontsize=16, fontweight='bold')
        plt.grid()
        plt.legend()

        if save:
            plt.savefig(f'plots/{title.replace(' ', '_')}.png')

        plt.show()



    



        

    
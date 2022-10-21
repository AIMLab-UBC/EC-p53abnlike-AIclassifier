import torch
import numpy as np
from utils.utils import print_mix

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Modify from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""
    def __init__(self,
                 patience: int = 10,
                 delta: int = 0,
                 lr_patience: int = 5) -> None:
        """
        Args:
            patience: int (default: 10)
                How long to wait after last time validation loss improved.
            delta: float (default: 0)
                Minimum change in the monitored quantity to qualify as an improvement.
            lr_patience: int (default: 5)
                How long to wait after last time validation loss improved to change the learning rate.
        """
        self.patience = patience
        self.counter = 0
        self.lr_patience = lr_patience
        self.best_score = None
        self.early_stop = False
        self.reduce_lr = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self,
                 val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # print_mix(f'EarlyStopping counter: __{self.counter}__ out of __{self.patience}__', 'RED')
            self.reduce_lr = False
            if self.counter == self.lr_patience:
                self.reduce_lr = True
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter    = 0
            self.reduce_lr  = False
            self.best_score = score

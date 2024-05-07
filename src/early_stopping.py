import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='../../model/region_BAE/transfer_adni/earlystopp_checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int) : how long to wait after last time validation loss imporced.
                             Default: 7
            verbose (bool) : If True, prints a message for each validation loss improvement.
                             Default: False
            delta (float) : Minimum change in the monitored quantity to qualify as an improvement.
                             Default: 0
            path (str) : Path for the checkpoint to be saved to.
            trace_func (function) : trace print function.
                             Default : print
        """
        self.patience = patience
        self.verbose = verbose
        self.count = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)

        elif score < self.best_score + self.delta:
            self.count += 1
            self.trace_func(f'EarlyStopping count: {self.counter} out of {self.patience}')
            if self.count >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, save_path):
        """Saves model with validation loss decreases"""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saviung model ...")
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
    


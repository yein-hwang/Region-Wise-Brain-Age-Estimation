import copy
import os
import pickle
import time
from pathlib import Path
from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm

from .utils import get_logger


class CNN_Trainer:
    def __init__(
        self,
        model,
        model_load_folder,
        model_save_folder,
        results_folder,
        dataloader_train,
        dataloader_valid,
        dataloader_test,
        epochs,
        optimizer,
        scheduler,
        cv_num,
        region,
        valid_ids,
        device,
        patience=0,
        logger=None,
    ):
        super().__init__()
        # Experiment logger: a wandb module when W&B is in use, otherwise a
        # no-op stand-in. Never None, so the log calls below always work.
        self.logger = logger if logger is not None else get_logger(use_wandb=False)
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_test = dataloader_test
        self.epoch = 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        # Early stopping: 0 disables; otherwise stop after `patience` epochs without valid-MAE improvement.
        self.patience = int(patience)

        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.cv_num = cv_num
        self.region = region
        self.model_load_folder = model_load_folder
        self.model_save_folder = Path(model_save_folder)
        self.model_save_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.train_mse_list, self.train_mae_list = [], []
        self.valid_mse_list, self.valid_mae_list = [], []
        self.valid_ids = valid_ids

        self.logger.watch(self.model, log='all')

    # ---- Private helpers ----

    def _move_batch_to_device(self, inputs, targets):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.reshape(-1, 1).to(self.device, non_blocking=True)
        return inputs, targets

    def _forward_features(self, inputs):
        if hasattr(self.model, 'module'):
            return self.model.module.forward_features(inputs)
        return self.model.forward_features(inputs)

    def _run_eval_loader(self, dataloader):
        mse_sum, mae_sum = 0.0, 0.0
        pred_ages_list, true_ages_list, feature_list = [], [], []

        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = self._move_batch_to_device(inputs, targets)
            outputs = self.model(inputs)
            features = self._forward_features(inputs)

            pred_ages_list.extend(outputs.detach().cpu().numpy().flatten().tolist())
            true_ages_list.extend(targets.detach().cpu().numpy().flatten().tolist())
            feature_list.append(features.detach().cpu())

            mse_loss = self.mse_loss_fn(outputs, targets)
            mae_loss = self.mae_loss_fn(outputs, targets)
            batch_size = inputs.size(0)
            mse_sum += mse_loss.item() * batch_size
            mae_sum += mae_loss.item() * batch_size

        n = len(dataloader.dataset)
        return mse_sum / n, mae_sum / n, pred_ages_list, true_ages_list, feature_list

    def _train_one_epoch(self):
        self.model.train()
        epoch_start = time.time()
        train_mse_sum, train_mae_sum = 0.0, 0.0

        for _, (inputs, targets) in enumerate(self.dataloader_train):
            inputs, targets = self._move_batch_to_device(inputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)

            mse_loss = self.mse_loss_fn(outputs, targets)
            mae_loss = self.mae_loss_fn(outputs, targets)

            mse_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = inputs.size(0)
            train_mse_sum += mse_loss.item() * batch_size
            train_mae_sum += mae_loss.item() * batch_size

        n = len(self.dataloader_train.dataset)
        train_mse_avg = train_mse_sum / n
        train_mae_avg = train_mae_sum / n

        self.train_mse_list.append(train_mse_avg)
        self.train_mae_list.append(train_mae_avg)

        self.logger.log({
            'Epoch': self.epoch + 1,
            'Learning rate': self.optimizer.param_groups[0]['lr'],
            'Train MSE Loss': train_mse_avg,
            'Train MAE Loss': train_mae_avg,
            'CV Split Number': self.cv_num,
        })

        print(f'Epoch {self.epoch + 1}: training duration {(time.time() - epoch_start) / 60:.2f} min')
        if os.getenv('REGBAE_MEASURE') == '1' and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f'Epoch {self.epoch + 1}: peak GPU MB {peak_mb:.1f}', flush=True)
            torch.cuda.reset_peak_memory_stats()
        return train_mse_avg, train_mae_avg

    def _validate(self):
        valid_start = time.time()
        self.model.eval()
        with torch.no_grad():
            valid_mse_avg, valid_mae_avg, pred_ages, true_ages, features = self._run_eval_loader(self.dataloader_valid)

        self.valid_mse_list.append(valid_mse_avg)
        self.valid_mae_list.append(valid_mae_avg)

        self.logger.log({
            'Epoch': self.epoch + 1,
            'Learning rate': self.optimizer.param_groups[0]['lr'],
            'Validation MSE Loss': valid_mse_avg,
            'Validation MAE Loss': valid_mae_avg,
        })

        print(f'Epoch {self.epoch + 1}: validation duration {(time.time() - valid_start) / 60:.2f} min')
        return valid_mse_avg, valid_mae_avg, pred_ages, true_ages, features

    # ---- Public methods ----

    def train(self):
        print('[ Start ]')
        best_mae = float('inf')
        best_epoch = -1
        best_pred_ages, best_true_ages, best_features = None, None, None
        no_improve_epochs = 0
        early_stopped = False
        total_start = time.time()

        for _ in tqdm(range(self.epochs)):
            train_mse_avg, train_mae_avg = self._train_one_epoch()
            valid_mse_avg, valid_mae_avg, pred_ages, true_ages, features = self._validate()

            self.save(self.epoch)

            # Track best model
            improved = valid_mae_avg < best_mae
            if improved:
                best_mae = valid_mae_avg
                best_epoch = self.epoch
                best_pred_ages = copy.deepcopy(pred_ages)
                best_true_ages = copy.deepcopy(true_ages)
                best_features = [f.clone() for f in features]
                no_improve_epochs = 0
                print(f'New best model at epoch {self.epoch + 1} with MAE: {best_mae:.4f}')
            else:
                no_improve_epochs += 1

            # Save last epoch results
            if self.epoch == self.epochs - 1:
                self._save_results(pred_ages, true_ages, features, valid_mae_avg)

            print(f'    Epoch {self.epoch + 1:2d}: train mse={train_mse_avg:.3f} / valid mse={valid_mse_avg:.3f}')
            print(f'    Epoch {self.epoch + 1:2d}: train mae={train_mae_avg:.3f} / valid mae={valid_mae_avg:.3f}')

            if self.dataloader_test is not None:
                self.test()

            self.logger.log({
                'Early Stop No-Improve Epochs': no_improve_epochs,
                'Best Validation MAE': best_mae,
                'Best Epoch': best_epoch + 1,
            })

            self.epoch += 1

            if self.patience > 0 and no_improve_epochs >= self.patience:
                early_stopped = True
                print(f'[ Early stopping ] No valid-MAE improvement for {self.patience} epochs '
                      f'(best epoch {best_epoch + 1}, MAE {best_mae:.4f}). Stopping at epoch {self.epoch}.')
                break

        print(f'[ End ] Total duration: {(time.time() - total_start) / 60:.2f} minutes')
        if early_stopped:
            self.logger.log({'Early Stopped': 1, 'Stopped Epoch': self.epoch})

        # Save best model results
        if best_pred_ages is not None:
            self._save_results(best_pred_ages, best_true_ages, best_features, best_mae, results_path='best')
            print(f'Best model from epoch {best_epoch + 1} with MAE: {best_mae:.4f}')

    def test(self):
        print('[ Start test ]')
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            test_mse_avg, test_mae_avg, pred_ages, true_ages, features = self._run_eval_loader(self.dataloader_test)

        self.logger.log({
            'Test MSE Loss': test_mse_avg,
            'Test MAE Loss': test_mae_avg,
            'Epoch': self.epoch + 1,
            'CV Split Number': self.cv_num,
        })

        print(f'test mse={test_mse_avg:.3f} / test mae={test_mae_avg:.3f}')
        self._save_results(pred_ages, true_ages, features, test_mae_avg, results_path='test')
        print(f'Duration for test: {(time.time() - start) / 60:.2f} minutes')

    def save(self, milestone):
        torch.save({
            'epoch': milestone + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_mse_list': self.train_mse_list,
            'train_mae_list': self.train_mae_list,
            'valid_mse_list': self.valid_mse_list,
            'valid_mae_list': self.valid_mae_list,
        }, self.model_save_folder / f'cv-{self.cv_num}-{milestone + 1}.pth.tar')

    def _find_best_epoch(self, cv_num, gpu):
        """Load the last checkpoint to find the epoch with lowest validation MAE."""
        folder = Path(self.model_load_folder)
        last_epoch = max(
            int(p.name.split('.')[0].split('-')[-1])
            for p in folder.glob(f'cv-{cv_num}-*.pth.tar')
        )
        last_path = folder / f'cv-{cv_num}-{last_epoch}.pth.tar'
        map_location = None if gpu else torch.device('cpu')
        ckpt = torch.load(last_path, map_location=map_location)
        mae_list = ckpt['valid_mae_list']
        best_epoch = int(min(range(len(mae_list)), key=lambda i: mae_list[i])) + 1
        print(f'Auto-detected best epoch: {best_epoch} (MAE: {mae_list[best_epoch - 1]:.4f}) from {last_path}')
        return best_epoch

    def load(self, cv_num, epoch='best', gpu=True):
        if epoch == 'best':
            epoch = self._find_best_epoch(cv_num, gpu)

        model_path = Path(self.model_load_folder) / f'cv-{cv_num}-{epoch}.pth.tar'

        if not model_path.exists():
            fallback = Path(self.model_load_folder) / f'cv-{cv_num}.pth.tar'
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f'No checkpoint found for cv={cv_num}, epoch={epoch}')

        map_location = None if gpu else torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=map_location)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(new_state_dict, strict=True)
        else:
            self.model.load_state_dict(new_state_dict, strict=True)

        self.train_mse_list, self.train_mae_list = [], []
        self.valid_mse_list, self.valid_mae_list = [], []

        print(f'============== Loaded weights from: {model_path}')

    def _save_results(self, pred_age_data, true_age_data, feature_data, test_mae_avg, results_path=''):
        results_folder = self.results_folder / str(self.cv_num)
        if results_path:
            results_folder = results_folder / results_path
        results_folder.mkdir(parents=True, exist_ok=True)

        save_file_name = results_folder / f'{self.region}.pkl'
        print(f'Save file: {save_file_name}')

        results_data = {
            'pred_ages': pred_age_data,
            'true_ages': true_age_data,
            'features': feature_data,
            'test_mae_avg': test_mae_avg,
        }

        if self.valid_ids is not None:
            results_data['valid_ids'] = self.valid_ids

        try:
            with open(save_file_name, 'wb') as file:
                pickle.dump(results_data, file)
            print(f'[INFO] Results saved at {save_file_name}', flush=True)
        except Exception as e:
            print(f'[ERROR] Failed to save results: {e}', flush=True)

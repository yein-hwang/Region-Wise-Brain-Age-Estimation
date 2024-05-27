import numpy as np
import pickle
import wandb
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from ewc import EWC

import time
import os


# Define trainer
class CNN_Trainer():
    def __init__(
            self, 
            model, 
            model_load_folder,
            model_save_folder,
            results_folder,
            dataloader_train, 
            dataloader_valid, 
            dataloader_test, 
            dataloader_train_old,
            epochs, 
            optimizer,
            early_stopping,
            scheduler,
            cv_num,
            importance):
        super(CNN_Trainer, self).__init__()

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_test = dataloader_test
        self.dataloader_train_old = dataloader_train_old
        self.epoch = 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.importance = importance
        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.cv_num = cv_num
        self.model_load_folder = model_load_folder
        self.model_save_folder = Path(model_save_folder)
        self.model_save_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.train_mse_list, self.train_mae_list = [], []
        self.valid_mse_list, self.valid_mae_list = [], []
        self.output_array, self.target_array = np.array([]), np.array([])
        self.feature_list = []

        wandb.watch(self.model, log="all")


    def train_ewc(self):

        # Load the fully trained model on the old dataset
        self.load(self.cv_num, 40) 
        # Initialize EWC with the fully trained model and the old data dataloader
        print("[ Initialize EWC ]")
        start = time.time()
        ewc = EWC(self.model, self.dataloader_train_old)
        end = time.time()  # End time
        # Compute the duration
        duration = (end - start) / 60
        print(f"Duration for Initialize EWC: {duration:.2f} minutes\n")


        self.model.train()        
        start = time.time()  # Start time
        print("[ Start ]")

        for i in tqdm(range(self.epochs)):
            print(f"\nEpoch {self.epoch+1:3d}: training")
            train_mse_sum, train_mae_sum, ewc_loss_sum, total_loss_sum = 0, 0, 0, 0
   
            for _, (input, target) in enumerate(self.dataloader_train):
                input = input.cuda(non_blocking=True)
                target = target.reshape(-1, 1)
                target = target.cuda(non_blocking=True)
                
                output = self.model(input)

                # ----------- update -----------
                self.optimizer.zero_grad()
                
                mae_loss = self.mae_loss_fn(output, target)
                mse_loss = self.mse_loss_fn(output, target)
                ewc_loss = ewc.penalty(self.model)
                total_loss = mse_loss + self.importance * ewc_loss
                total_loss.backward() 
                self.optimizer.step()
                self.scheduler.step()
                
                # Logging to wandb
                wandb.log({
                    "Learning rate": self.optimizer.param_groups[0]['lr'],
                })

                train_mse_sum += mse_loss.item() * input.size(0)
                train_mae_sum += mae_loss.item() * input.size(0)
                ewc_loss_sum += ewc_loss.item() * input.size(0)
                total_loss_sum += total_loss.item() * input.size(0)

            train_mse_avg = train_mse_sum / len(self.dataloader_train.dataset)
            train_mae_avg = train_mae_sum / len(self.dataloader_train.dataset)
            ewc_loss_avg = ewc_loss_sum / len(self.dataloader_train.dataset)
            total_loss_avg = total_loss_sum / len(self.dataloader_train.dataset)

            self.train_mse_list.append(train_mse_avg)
            self.train_mae_list.append(train_mae_avg)
            
            wandb.log({
                "Epoch": self.epoch + 1,
                "Learning rate": self.optimizer.param_groups[0]['lr'],
                "Train MSE Loss": train_mse_avg,
                "Train MAE Loss": train_mae_avg,
                "EWC Loss": ewc_loss_avg,
                "Total Loss": total_loss_avg,
                "CV Split Number": self.cv_num
            })
            
            end = time.time()  # End time
            # Compute the duration
            duration = (end - start) / 60
            print(f"Epoch: {self.epoch+1}, duration for training: {duration:.2f} minutes")


            # ==================== validation step
            print(f"\nEpoch {self.epoch+1:3d}: validation")
            start = time.time()  # Start time
            self.model.eval()
            with torch.no_grad():
                valid_mse_sum, valid_mae_sum = 0, 0
                for _, (input, target) in enumerate(self.dataloader_valid):
                    input = input.cuda(non_blocking=True)
                    target = target.reshape(-1, 1)
                    target = target.cuda(non_blocking=True)

                    output = self.model(input)

                    mse_loss = self.mse_loss_fn(output, target) 
                    mae_loss = self.mae_loss_fn(output, target)

                    valid_mse_sum += mse_loss.item()*input.size(0) # mse_loss.item() * input.size(0): 각 배치에서의 총 손실을 계산
                                                                   # input.size(0): 배치 내의 샘플 수를 반환, 이것을 MSE 손실 값에 곱하면 해당 배치의 전체 손실
                                                                   # validation set의 모든 배치를 통해 계산된 총 손실을 더하여 합산
                    valid_mae_sum += mae_loss.item()*input.size(0)

                valid_mse_avg = valid_mse_sum / len(self.dataloader_valid.dataset)
                valid_mae_avg = valid_mae_sum / len(self.dataloader_valid.dataset)

                self.valid_mse_list.append(valid_mse_avg)
                self.valid_mae_list.append(valid_mae_avg)

                # learning rate scheduler update
                if self.scheduler:
                    self.scheduler.step(valid_mse_avg)

                # early stopping related
                if self.early_stopping is None:
                    # Save each model in every epoch
                    self.save(self.epoch)
                else:
                    save_path = f"{self.model_save_folder}/early_stop/cv-{self.cv_num}-{self.epoch+1}.pth.tar"
                    self.early_stopping(valid_mae_avg, self.model, self.epoch, save_path)
                    if self.early_stopping.early_stop:
                        break


                wandb.log({
                    "Epoch": self.epoch+1,
                    "Learning rate": self.optimizer.param_groups[0]['lr'],
                    "Validation MSE Loss": valid_mse_avg,
                    "Validation MAE Loss": valid_mae_avg
                })


                print(f"    Epoch {self.epoch+1:2d}: training mse loss = {train_mse_avg:.3f} / validation mse loss = {valid_mse_avg:.3f}")
                print(f"    Epoch {self.epoch+1:2d}: training mae loss = {train_mae_avg:.3f} / validation mae loss = {valid_mae_avg:.3f}")
                    
                
            self.epoch += 1
            
        print("[ End of Epoch ]")
        end = time.time()  # End time
        # Compute the duration and GPU usage
        duration = (end - start) / 60
        print(f"Epoch: {self.epoch}, duration for validation: {duration:.2f} minutes")
        
        return self.train_mse_list, self.train_mae_list, self.valid_mse_list, self.valid_mae_list
    
    
    def test(self):
        print("[ Start test ]")

        self.load(self.cv_num, 40, True) # pre-trained model load
        self.model.eval()
        with torch.no_grad():
            test_mse_sum, test_mae_sum = 0, 0
            pred_ages_list = []
            true_ages_list = []

            for _, (input, target) in enumerate(tqdm(self.dataloader_test)):
                input = input.cuda(non_blocking=True)
                target = target.reshape(-1, 1)
                target = target.cuda(non_blocking=True)

                output = self.model(input)

                # feature extraction
                features = self.model.module.forward_features(input) 
                self.feature_list.append(features.cpu())  # CPU로 옮겨 리스트에 추가

                pred_ages = output.cpu().numpy()
                true_ages = target.cpu().numpy()
                pred_ages_list.extend(pred_ages.flatten())
                true_ages_list.extend(true_ages.flatten())

                mse_loss = self.mse_loss_fn(output, target) 
                mae_loss = self.mae_loss_fn(output, target)
                
                test_mse_sum += mse_loss.item()*input.size(0)
                test_mae_sum += mae_loss.item()*input.size(0)

            test_mse_avg = test_mse_sum / len(self.dataloader_test.dataset)
            test_mae_avg = test_mae_sum / len(self.dataloader_test.dataset)
            
            print(f"test mse loss = {test_mse_avg:.3f} / test mae loss = {test_mae_avg:.3f}")
        
        return pred_ages_list, true_ages_list, self.feature_list

            
    
    def save(self, milestone):
        model_save_directory = Path(f"{self.model_save_folder}/cv-{self.cv_num}")
        model_save_directory.mkdir(parents=True, exist_ok=True)
        save_path = model_save_directory / f"model_epoch_{milestone+1}.pth.tar"
        torch.save({
            "epoch": milestone + 1, 
            "state_dict": self.model.state_dict(), 
            "optimizer": self.optimizer.state_dict(),  
            "train_mse_list": self.train_mse_list,
            "train_mae_list": self.train_mae_list,
            "valid_mse_list": self.valid_mse_list,
            "valid_mae_list": self.valid_mae_list
        }, save_path)
        
    def load(self, cv_num, epoch, test=False):
        if test:
            model_path = f'{self.model_load_folder}/model_epoch_{epoch}.pth.tar'
        else:
            model_path = f'{self.model_load_folder}/cv-{cv_num}-{epoch}.pth.tar'
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_mse_list = checkpoint.get("train_mse_list", [])
        self.train_mae_list = checkpoint.get("train_mae_list", [])
        self.valid_mse_list = checkpoint.get("valid_mse_list", [])
        self.valid_mae_list = checkpoint.get("valid_mae_list", [])

        print(f"============== Loaded model: {model_path}")
        for name, param in self.model.named_parameters():
            print(f"{name}: mean = {param.mean().item()}, std = {param.std().item()}")

    
    def age_data_extraction(self, pred_age_data, true_age_data, feature_data):

        pred_ages_filename = os.path.join(self.results_folder, 'pred_ages.pkl')
        true_ages_filename = os.path.join(self.results_folder, 'true_ages.pkl')
        features_filename = os.path.join(self.results_folder, 'features.pkl')
        print(self.results_folder)

        try:
            with open(pred_ages_filename, 'wb') as file:
                pickle.dump(pred_age_data, file)
            with open(true_ages_filename, 'wb') as file:
                pickle.dump(true_age_data, file)
            print(f"Ages saved")
            with open(features_filename, 'wb') as file:
                pickle.dump(feature_data, file)
            print(f"Features saved")
        except Exception as e:
            print("Error saving data:", e)
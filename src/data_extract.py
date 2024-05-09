import pandas as pd
import numpy as np
import glob
import os
import time
from tqdm import tqdm
from pathlib import Path
import math
import wandb
import gc
import pickle

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary
from torch import nn
from torchvision import transforms as T
import nibabel as nib
from skimage.transform import resize

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)
gc.collect()
wandb.init()


# Model define
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            
# Define 3D_CNN model class

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96)
        )
        
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(768, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.flatten(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

# Define trainer
class CNN_Trainer():
    def __init__(
            self, 
            model, 
            results_folder, 
            dataloader_train, 
            dataloader_valid, 
            dataloader_test, 
            epochs, 
            optimizer,
            scheduler,
            cv_num,
            model_load):
        super(CNN_Trainer, self).__init__()

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_test = dataloader_test
        self.epoch = 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.cv_num = cv_num
        self.model_load = model_load
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        print(self.results_folder)
        
        self.train_mse_list, self.train_mae_list = [], []
        self.valid_mse_list, self.valid_mae_list = [], []
        self.feature_list = []

        wandb.watch(self.model, log="all")

    def train(self):
        print("[ Start ]")
        
        if self.model_load == 1:
            # Load the model
            self.load(self.cv_num)

        self.model.train()
        
        start = time.time()  # Start time
        # while self.epoch < self.epochs
        for i in tqdm(range(self.epochs)):
            print(f"\nEpoch {self.epoch+1:3d}: training")
            train_mse_sum, train_mae_sum = 0, 0
            # for batch_ID, (input, target) in enumerate(tqdm(self.dataloader_train)):
            for batch_ID, (input, target) in enumerate(self.dataloader_train):
                input = input.cuda(non_blocking=True)
                target = target.reshape(-1, 1)
                target = target.cuda(non_blocking=True)
                
                output = self.model(input)

                # ----------- update -----------
                self.optimizer.zero_grad()

                mse_loss = self.mse_loss_fn(output, target)
                mae_loss = self.mae_loss_fn(output, target)
                
                mse_loss.backward() # loss_fn should be the one used for backpropagation
                self.optimizer.step()
                self.scheduler.step()
                
                wandb.log({
                "Learning rate": self.optimizer.param_groups[0]['lr'],
                })

                train_mse_sum += mse_loss.item()*input.size(0)
                train_mae_sum += mae_loss.item()*input.size(0)

            train_mse_avg = train_mse_sum / len(self.dataloader_train.dataset)
            train_mae_avg = train_mae_sum / len(self.dataloader_train.dataset)

            self.train_mse_list.append(train_mse_avg)
            self.train_mae_list.append(train_mae_avg)
            
            wandb.log({
                "Epoch": self.epoch+1,
                "Learning rate": self.optimizer.param_groups[0]['lr'],
                "Train MSE Loss": train_mse_avg,
                "Train MAE Loss": train_mae_avg,
                "CV Split Number": self.cv_num
            })
            
            end = time.time()  # End time
            # Compute the duration
            duration = (end - start) / 60
            print(f"Epoch: {self.epoch+1}, duration for training: {duration:.2f} minutes")

            # ==================== validation step
            print(f"\nEpoch {self.epoch+1:3d}: validation")
            start = time.time()  # Start time
            # self.model.eval()
            with torch.no_grad():
                valid_mse_sum, valid_mae_sum = 0, 0
                for _, (input, target) in enumerate(tqdm(self.dataloader_valid)):
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
                
                self.scheduler.step(valid_mse_avg)
                print(f"    Epoch {self.epoch+1:2d}: training mse loss = {train_mse_avg:.3f} / validation mse loss = {valid_mse_avg:.3f}")
                print(f"    Epoch {self.epoch+1:2d}: training mae loss = {train_mae_avg:.3f} / validation mae loss = {valid_mae_avg:.3f}")
                

                # Save each model in every epoch
                self.save(self.epoch)
                    
                wandb.log({
                    "Epoch": self.epoch+1,
                    "Learning rate": self.optimizer.param_groups[0]['lr'],
                    "Validation MSE Loss": valid_mse_avg,
                    "Validation MAE Loss": valid_mae_avg
                })
                
            self.epoch += 1
            
        print("[ End of Epoch ]")
        end = time.time()  # End time
        # Compute the duration and GPU usage
        duration = (end - start) / 60
        print(f"Epoch: {self.epoch}, duration for validation: {duration:.2f} minutes")
        
        return self.train_mse_list, self.train_mae_list, self.valid_mse_list, self.valid_mae_list
    
    
    def test(self):
        print("[ Start test ]", self.dataloader_test)
        with torch.no_grad():
            test_mse_sum, test_mae_sum = 0, 0
            # pred_age and true age 저장할 리스트
            pred_ages = []
            true_ages = []

            
            for _, (input, target) in enumerate(tqdm(self.dataloader_test)):
                input = input.cuda(non_blocking=True)
                target = target.reshape(-1, 1)
                target = target.cuda(non_blocking=True)

                output = self.model(input)
                
                # feature extraction
                features = self.model.module.forward_features(input)  # 특징 추출
                self.feature_list.append(features.cpu())  # CPU로 옮겨 리스트에 추가
                
                # current age values saving
                # pred_ages = output.cpu().numpy().flatten()
                # true_ages = target.cpu().numpy().flatten()
                pred_age = output.cpu().numpy()[0, 0]
                true_age = target.cpu().numpy()[0, 0]
                pred_ages.append(pred_age)
                true_ages.append(true_age)

                mse_loss = self.mse_loss_fn(output, target) 
                mae_loss = self.mae_loss_fn(output, target)
                
                test_mse_sum += mse_loss.item()*input.size(0)
                test_mae_sum += mae_loss.item()*input.size(0)

            test_mse_avg = test_mse_sum / len(self.dataloader_test.dataset)
            test_mae_avg = test_mae_sum / len(self.dataloader_test.dataset)
            
            print(f"test mse loss = {test_mse_avg:.3f} / test mae loss = {test_mae_avg:.3f}")

            wandb.log({
                "Test MSE Loss": test_mse_avg,
                "Test MAE Loss": test_mae_avg
            })

        return pred_ages, true_ages, feature_list
            
    def save_features(self, milestone):
        all_features = torch.cat(self.feature_list, dim=0)
        # features 디렉토리 경로 설정
        features_folder = Path(self.results_folder) / 'feat'
        features_folder.mkdir(parents=True, exist_ok=True)
        torch.save(all_features, f"{features_folder}/cv-{self.cv_num}-features-{milestone+1}.pt")

    def load_features(self, milestone):
        # 저장된 특징을 로드
        features_path = f"{self.results_folder}/cv-{self.cv_num}-features-{milestone+1}.pt"
        all_features = torch.load(features_path)
        return all_features
    
    def save(self, milestone):
        torch.save({"epoch": milestone+1, 
                    "state_dict": self.model.state_dict(), 
                    "optimizer" : self.optimizer.state_dict(),  
                    "train_mse_list": self.train_mse_list,
                    "train_mae_list": self.train_mae_list,
                    "valid_mse_list": self.valid_mse_list,
                    "valid_mae_list": self.valid_mae_list},  
                    f"{self.results_folder}/cv-{self.cv_num}-{milestone+1}.pth.tar")
        # self.save_features(milestone)
        
    def tf_load(self, milestone):
        # region_name = os.path.basename(self.results_folder) # region_name extraction
        # model_root = os.path.join(os.path.dirname(os.path.dirname(self.results_folder)), region_name)
        model_path = f'{self.results_folder}/cv-{milestone}-40.pth.tar'
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_mse_list = checkpoint.get("train_mse_list", [])
        self.train_mae_list = checkpoint.get("train_mae_list", [])
        self.valid_mse_list = checkpoint.get("valid_mse_list", [])
        self.valid_mae_list = checkpoint.get("valid_mae_list", [])

        print(f"============== Loaded model: {model_path}")

    
    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]  # Get the epoch directly from the checkpoint
        self.train_mse_list = checkpoint.get("train_mse_list", [])
        self.train_mae_list = checkpoint.get("train_mae_list", [])
        self.valid_mse_list = checkpoint.get("valid_mse_list", [])
        self.valid_mae_list = checkpoint.get("valid_mae_list", [])


class Region_Dataset(Dataset):
    def __init__(self, dataset_df, indices=None, roi=None):
        super(Region_Dataset, self).__init__()
        self.data_csv = dataset_df
        self.roi = roi
        
        self.image_paths = [self.data_csv[self.roi][i] for i in indices]
        self.ages = [self.data_csv['age'][i] for i in indices]
    
        self.transform = T.Compose([
            T.ToTensor()
        ])
    def collate_fn(self, batch):
        images, labels = zip(*batch)  # separate images and labels
        images = torch.stack(images)  # stack images into a tensor
        ages = torch.tensor(labels)  # convert labels into a tensor
        return images, ages

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        age = self.ages[index]

        # Load the image and convert to a tensor
        image = nib.load(image_path).get_fdata()
        image = torch.from_numpy(image).float()

        # 채널 차원 추가
        image = image.unsqueeze(0)  # 첫 번째 차원에 채널 추가

        age = torch.tensor(age, dtype=torch.float32)

        return (image, age)


class CustomCosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CustomCosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


ukb_df = pd.read_csv('/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/ukbb_cn_region.csv')
DATA_SIZE = len(ukb_df)
regions = {0: 'imgs', 1: 'caudate', 2: 'cerebellum', 3: 'frontal_lobe', 4: 'insula', 5: 'occipital_lobe', 6: 'parietal_lobe', 7: 'putamen', 8: 'temporal_lobe', 9: 'thalamus'}


for i in regions.keys():
    ROI = regions[i]
    print("=" * 20, ROI, "=" * 20)
    dataset_indices = list(ukb_df.index)[:DATA_SIZE]
    test_dataset = Region_Dataset(ukb_df, dataset_indices, ROI)
    dataloader_test = DataLoader(test_dataset, 
                                batch_size=1, 
                                sampler=SequentialSampler(test_dataset),
                                collate_fn=test_dataset.collate_fn,
                                pin_memory=True,
                                num_workers=2)
    
    # hypterparameters
    BATCH_SIZE = 4
    EPOCHS = 2
    RESULTS_FOLDER = './test'
    INPUT_SIZE = (1, 128, 128, 128)
    LEARNING_RATE = 1e-6
    N_WORKERS = 8
    
    # Initialize your model (Make sure it's the same architecture as the one you trained)
    model = CNN(in_channels=1).cuda()  
    # Put your model on the GPU
    model = torch.nn.DataParallel(model)
    
    # Define your optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    t_0 = int(DATA_SIZE * 0.75 // BATCH_SIZE // 6)
    scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer,T_0= t_0, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)
    
    # Loss function
    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    
    # Initialize the trainer
    trainer = CNN_Trainer(
        model=model, 
        results_folder=RESULTS_FOLDER,  # replace with the path where your results are saved
        dataloader_train=None,  # You don't need the training set for inference
        dataloader_valid=None,  # You don't need the validation set for inference
        dataloader_test=dataloader_test, 
        epochs=0,  # You don't need to specify epochs for inference
        optimizer=optimizer,
        scheduler=scheduler,
        cv_num=0,  # replace with the appropriate iteration number
        model_load=1
    )

    # model loading
    pred_ages_lists = []
    true_ages_lists = []
    features_lists = []

    model_root = f'/media/leelabsg-storage1/yein/research/model/region_BAE/{ROI}' 
    for cv_num in range(4):
        model_path = f'{model_root}/cv-{cv_num}-40.pth.tar'
        print(model_path)
        checkpoint = torch.load(model_path, map_location='cuda')
        trainer.load(checkpoint)
        pred_ages, true_ages, features = trainer.test()

        pred_ages_lists.append(pred_agees)
        true_ages_lists.append(true_ages)
        features_lists.append(features)

    # data 저장
    ages_path = f'/media/leelabsg-storage1/yein/research/test/ukb/ages/{ROI}'
    features_path = f'/media/leelabsg-storage1/yein/research/test/ukb/features/{ROI}'
    if not os.path.exists(ages_path):
        os.makedirs(ages_path)
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    
    # pred_ages_lists와 true_ages_lists를 저장합니다.
    ages_filename = os.path.join(ages_path, 'ages_lists.pkl')
    with open(ages_filename, 'wb') as file:
        pickle.dump({'pred_ages': pred_ages_lists, 'true_ages': true_ages_lists}, file)
        print(f"Ages saved to {ages_filename}")
    
    # features_lists를 저장합니다.
    features_filename = os.path.join(features_path, 'features_lists.pkl')
    with open(features_filename, 'wb') as file:
        pickle.dump(features_lists, file)
        print(f"Features saved to {features_filename}")
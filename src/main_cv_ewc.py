import time
import pickle
import random
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import KFold

from config_ewc import Config, parse_args
from dataset import *
from CNN import *
from CNN_Trainer_ewc import *
from learning_rate import lr_scheduler as lr
from early_stopping import EarlyStopping

import gc

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary

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

args = parse_args()
config = Config(args)

# setting & hypterparameters
IMGS_o = config.mri_csv_old 
IMGS_n = config.mri_csv_new
dataset_df_o = pd.read_csv(IMGS_o)
dataset_df_n = pd.read_csv(IMGS_n)
DATA_SIZE_o = len(dataset_df_o)
DATA_SIZE_n = len(dataset_df_n)

BATCH_SIZE = config.batch_size
EPOCHS = config.nb_epochs
RESULTS_FOLDER = config.results_folder
MODEL_SAVE_FOLDER = config.model_save_folder
INPUT_SIZE = config.input_size
LEARNING_RATE = config.lr
N_WORKERS = config.num_cpu_workers
REGIONS = config.regions
MODEL_LOAD_FOLDER = config.model_load_folder
MODEL_LOAD = 1
MODEL_LOAD_EPOCH = config.model_load_epoch
MODE = config.mode
PATIENCE = 0
# PATIENCE = config.patience
IMPORTANCE = config.importance

# setting log
print("="* 20, " Setting ", "="* 20)
# Check the number of gpus
ngpus = torch.cuda.device_count()

print("Mode :                   ", MODE)
print("Number of gpus :         ", ngpus)
print("Batch size :             ", BATCH_SIZE)
print("Data A size:             ", DATA_SIZE_o)
print("Data B size:             ", DATA_SIZE_n)
print("Epochs :                 ", EPOCHS)
print("Importance(Lambda) :     ", IMPORTANCE)
print("Early Stopping Patience :", PATIENCE)
print("# of Workers  :          ", N_WORKERS)
print("="* 50)

wandb_path = "/media/leelabsg-storage1/yein/research/wandb/RegionBAE"
wandb.init(dir=wandb_path, project="reg_trial", settings=wandb.Settings(start_method="fork"))


# create our k-folds
kf = KFold(n_splits=4, random_state=7, shuffle=True)
# obtain the indices for each dataset
dataset_indices_old = list(dataset_df_o.index)[:DATA_SIZE_o]
dataset_indices_new = list(dataset_df_n.index)[:DATA_SIZE_n]
train_indices_old, valid_indices_old, train_indices_new, valid_indices_new = [], [], [], []
for train_idx_o, valid_idx_o in kf.split(dataset_indices_old):
    train_indices_old.append(train_idx_o)
    valid_indices_old.append(valid_idx_o)
for train_idx_n, valid_idx_n in kf.split(dataset_indices_new):
    train_indices_new.append(train_idx_n)
    valid_indices_new.append(valid_idx_n)

for _, ROI in REGIONS.items():  
    #add wandb run name
    proj_n = f"ewc_{ROI}_{IMPORTANCE}"
    wandb.run.name = proj_n
    MODEL_LOAD_FOLDER = os.path.join(MODEL_LOAD_FOLDER, ROI)
    MODEL_SAVE_FOLDER = os.path.join(MODEL_SAVE_FOLDER, ROI)
    print("Model Load Path:         ", MODEL_LOAD_FOLDER)
    print("Model Save Path:         ", MODEL_SAVE_FOLDER)
    
    for cv_num in range(4):
        print('\n<<< StratifiedKFold: {0}/{1} >>>'.format(cv_num+1, 4))
        # ------------------------ Train the model
        if MODE == 'train': 

            # create a old train dataset for this fold
            train_dataset_old = Region_Dataset(config.old_root, dataset_df_o, train_indices_old[cv_num], ROI)
            dataloader_train_old = DataLoader(train_dataset_old, 
                                        batch_size=BATCH_SIZE, 
                                        sampler=RandomSampler(train_dataset_old),
                                        collate_fn=train_dataset_old.collate_fn,
                                        pin_memory=True,
                                        num_workers=N_WORKERS)

            # create a new dataset for this fold
            train_dataset = Region_Dataset(config.new_root, dataset_df_n, train_indices_new[cv_num], ROI)
            valid_dataset = Region_Dataset(config.new_root, dataset_df_n, valid_indices_new[cv_num], ROI)
            
            dataloader_train = DataLoader(train_dataset, 
                                        batch_size=BATCH_SIZE, 
                                        sampler=RandomSampler(train_dataset),
                                        collate_fn=train_dataset.collate_fn,
                                        pin_memory=True,
                                        num_workers=N_WORKERS)
            dataloader_valid = DataLoader(valid_dataset, 
                                        batch_size=BATCH_SIZE, 
                                        sampler=SequentialSampler(valid_dataset),
                                        collate_fn=valid_dataset.collate_fn,
                                        pin_memory=True,
                                        num_workers=N_WORKERS)

            # Define model and optimizer
            model = CNN(in_channels=1).cuda()
            # Apply the weight_initialiation
            model.apply(initialize_weights)
            model = torch.nn.DataParallel(model) # use with multi-gpu environment
            # summary(model, input_size=INPUT_SIZE, device="cuda") # model-summary
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            t_0 = int(len(train_indices_new[cv_num]) // BATCH_SIZE // 6)
            scheduler = lr.CustomCosineAnnealingWarmUpRestarts(optimizer,T_0= t_0, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)
            
            # Early Stopping
            if PATIENCE == 0:
                EARLY_STOPPING = None
            else:
                EARLY_STOPPING = EarlyStopping(patience=config.patience, verbose=True)

            trainer = CNN_Trainer(model=model, 
                                model_load_folder=MODEL_LOAD_FOLDER,
                                model_save_folder=MODEL_SAVE_FOLDER,
                                results_folder=RESULTS_FOLDER,
                                dataloader_train=dataloader_train, 
                                dataloader_valid=dataloader_valid, 
                                dataloader_test=None,
                                dataloader_train_old=dataloader_train_old, 
                                epochs=EPOCHS, 
                                optimizer=optimizer, 
                                early_stopping=EARLY_STOPPING,
                                scheduler=scheduler,
                                cv_num=cv_num,
                                importance=IMPORTANCE)
        
            train_start = time.time()
            trainer.train_ewc() # This will create the lists as instance variables

            # Now you can access the lists as:
            train_mse_list = trainer.train_mse_list
            train_mae_list = trainer.train_mae_list
            valid_mse_list = trainer.valid_mse_list
            valid_mae_list = trainer.valid_mae_list

            print("train_mse_list: ", train_mse_list)
            print("train_mae_list: ", train_mae_list)
            print("valid_mse_list: ", valid_mse_list)
            print("valid_mae_list: ", valid_mae_list)
            train_end = time.time()

            print(f"\nElapsed time for one split in cv: {(train_end - train_start) / 60:.0f} minutes")
    
    # # ------------------------ Test the model
    # else:
        
    #     pred_age_data = dict()
    #     true_age_data = dict()
    #     feature_data = dict()

    #     results_folder = ''
    
    #     for _, v in REGIONS.items():      

    #         # model_save folder & results folder related
    #         if MODE == 'test':
    #             model_save_folder = os.path.join(MODEL_SAVE_FOLDER, v)  
    #             results_folder = os.path.join(RESULTS_FOLDER, str(cv_num))
    #         else: # MODE == 'test_tf'
    #             model_save_folder = os.path.dirname(MODEL_SAVE_FOLDER)
    #             model_save_folder = os.path.join(model_save_folder, 'transfer_adni', v)
    #             results_folder = os.path.join(RESULTS_FOLDER + '_tf', str(cv_num))

    #         # test_dataset related
    #         if DATASET == 'adni' and MODE == 'test':
    #             test_dataset = Region_Dataset(config, dataset_indices, v)
    #         else:
    #             test_dataset = Region_Dataset(config, valid_indices, v)
    #         dataloader_test = DataLoader(test_dataset, 
    #                                     batch_size=BATCH_SIZE, 
    #                                     sampler=SequentialSampler(test_dataset),
    #                                     collate_fn=test_dataset.collate_fn,
    #                                     pin_memory=True,
    #                                     num_workers=N_WORKERS)

    #         trainer = CNN_Trainer(model=model, 
    #                             model_load_folder = MODEL_LOAD_FOLDER,
    #                             model_save_folder=model_save_folder,
    #                             results_folder=results_folder,
    #                             dataloader_train=None, 
    #                             dataloader_valid=None, 
    #                             dataloader_test=dataloader_test,
    #                             epochs=EPOCHS, 
    #                             optimizer=optimizer, 
    #                             early_stopping=None,
    #                             scheduler=scheduler,
    #                             cv_num=cv_num,
    #                             model_load=MODEL_LOAD)

    #         trainer.load(cv_num, MODEL_LOAD_EPOCH) # pre-trained model load
    #         pred_ages, true_ages, features = trainer.test() # test

    #         pred_age_data.setdefault(v, [])
    #         true_age_data.setdefault(v, [])
    #         feature_data.setdefault(v, [])

    #         pred_age_data[v].extend(pred_ages)
    #         true_age_data[v].extend(true_ages)
    #         feature_data[v].extend(features)

    #     # save the data
    #     trainer.age_data_extraction(pred_age_data,
    #                                 true_age_data,
    #                                 feature_data)

    #     # pred_ages_filename = os.path.join(results_folder, 'pred_ages.pkl')
    #     # true_ages_filename = os.path.join(results_folder, 'true_ages.pkl')
    #     # features_filename = os.path.join(results_folder, 'features.pkl')
    #     # print(results_folder)

    #     # try:
    #     #     with open(pred_ages_filename, 'wb') as file:
    #     #         pickle.dump(pred_age_data, file)
    #     #     with open(true_ages_filename, 'wb') as file:
    #     #         pickle.dump(true_age_data, file)
    #     #     print(f"Ages saved")
    #     #     with open(features_filename, 'wb') as file:
    #     #         pickle.dump(feature_data, file)
    #     #     print(f"Features saved")
    #     # except Exception as e:
    #     #     print("Error saving data:", e)







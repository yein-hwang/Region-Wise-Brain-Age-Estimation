import time
import pickle
import random
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import KFold

from config import Config, parse_args
from dataset import *
from CNN import *
from CNN_Trainer import *
from learning_rate import lr_scheduler as lr
from early_stopping import EarlyStopping
from ewc import EWC

import gc

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(0)

gc.collect()

args = parse_args()
config = Config(args)

# setting & hypterparameters
IMGS = config.mri_csv # preprocessed global and 9 regions mri paths
DATASET = config.dataset
dataset_df = pd.read_csv(IMGS)
BATCH_SIZE = config.batch_size
EPOCHS = config.nb_epochs
RESULTS_FOLDER = config.results_folder
MODEL_SAVE_FOLDER = config.model_save_folder
INPUT_SIZE = config.input_size
LEARNING_RATE = config.lr
LEARNING_RATE_Scheduler = config.lr_scheduler_choice
N_WORKERS = config.num_cpu_workers
REGIONS = config.regions
ROI = config.roi
MODEL_LOAD_FOLDER = config.model_load_folder
MODEL_LOAD = config.model_load
MODEL_LOAD_EPOCH = config.model_load_epoch
# DATA_SIZE = config.data_size
DATA_SIZE = len(dataset_df)
MODE = config.mode
PATIENCE = config.patience

# setting log
print("="* 20, " Setting ", "="* 20)
# Check the number of gpus
ngpus = torch.cuda.device_count()

print("Dataset :                 ", DATASET)
print("Mode :                    ", MODE)
print("Number of gpus :          ", ngpus)
print("Batch size :             ", BATCH_SIZE)
print("Data size:               ", DATA_SIZE)
print("Epochs :                 ", EPOCHS)
print("Learning Rate :          ", LEARNING_RATE)
print("Early Stopping Patience :", PATIENCE)
print("# of Workers  :          ", N_WORKERS)
print("Region of Interest :     ", ROI)
print("Model Save Path:         ", MODEL_SAVE_FOLDER)
print("Loaded Model Epoch:      ", MODEL_LOAD_EPOCH)
print("="* 50)

wandb_path = "/media/leelabsg-storage1/yein/research/wandb/RegionBAE"
wandb.init(dir=wandb_path, project="reg_trial", settings=wandb.Settings(start_method="fork"))
#add wandb run name
proj_n = f"{ROI}_{DATA_SIZE}_{EPOCHS}"
wandb.run.name = proj_n


# create our k-folds
kf = KFold(n_splits=4, random_state=7, shuffle=True)
cv_num = 0
# obtain the indices for our dataset
dataset_indices = list(dataset_df.index)[:DATA_SIZE]

# loop over each fold
for train_indices, valid_indices in kf.split(dataset_indices):

    print('\n<<< StratifiedKFold: {0}/{1} >>>'.format(cv_num+1, 4))
    
    # create a new dataset for this fold
    # train_dataset = Region_Dataset(config.root, dataset_df, train_indices, ROI)
    # valid_dataset = Region_Dataset(config.root, dataset_df, valid_indices, ROI)
    train_dataset = Region_Dataset(dataset_df, train_indices, ROI)
    valid_dataset = Region_Dataset(dataset_df, valid_indices, ROI)    
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
    
    print("Train Dataset & Validation Dataset size: ", len(dataloader_train), len(dataloader_valid))
    print(valid_indices)

    # Define model and optimizer
    model = CNN(in_channels=1).cuda()
    # Apply the weight_initialiation
    model.apply(initialize_weights)
    model = torch.nn.DataParallel(model) # use with multi-gpu environment
    # summary(model, input_size=INPUT_SIZE, device="cuda") # model-summary

    if LEARNING_RATE_Scheduler == 0:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config.weight_decay)
        scheduler = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        t_0 = int(len(train_indices) // BATCH_SIZE // 6)
        scheduler = lr.CustomCosineAnnealingWarmUpRestarts(optimizer,T_0= t_0, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)

    # Loss function
    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    
    # ------------------------ Train the model
    if MODE == 'train': 

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
                            epochs=EPOCHS, 
                            optimizer=optimizer, 
                            early_stopping=EARLY_STOPPING,
                            scheduler=scheduler,
                            cv_num=cv_num,
                            region=ROI,
                            model_load=MODEL_LOAD)
        # Model Loading
        if MODEL_LOAD == 1:
            trainer.load(cv_num, MODEL_LOAD_EPOCH) # pre-trained model load
            
        train_start = time.time()
        trainer.train() # This will create the lists as instance variables

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

        print(f"\nElapsed time for one epoch in cv: {(train_end - train_start) / 60:.0f} minutes")
    
    # ------------------------ Test the model
    else:
        
        pred_age_data = dict()
        true_age_data = dict()
        feature_data = dict()

        results_folder = ''
    
        for _, v in REGIONS.items():      

            model_load_folder = os.path.join(MODEL_LOAD_FOLDER, v)

            # model_save folder & results folder related
            if MODE == 'test':
                results_folder = os.path.join(RESULTS_FOLDER, str(cv_num))
            else: # MODE == 'test_tf'
                results_folder = os.path.join(RESULTS_FOLDER + '_tf', str(cv_num))

            # test_dataset related
            test_dataset = Region_Dataset(dataset_df, dataset_indices, v)
            dataloader_test = DataLoader(test_dataset, 
                                        batch_size=BATCH_SIZE, 
                                        sampler=SequentialSampler(test_dataset),
                                        collate_fn=test_dataset.collate_fn,
                                        pin_memory=True,
                                        num_workers=N_WORKERS)
            if (DATASET == 'ukbb' and MODE == 'test') or (DATASET == 'adni' and MODE == 'test_tf'):
                dataloader_test = dataloader_valid

            trainer = CNN_Trainer(model=model, 
                                model_load_folder = model_load_folder,
                                model_save_folder=MODEL_SAVE_FOLDER,
                                results_folder=results_folder,
                                dataloader_train=None, 
                                dataloader_valid=None, 
                                dataloader_test=dataloader_test,
                                epochs=EPOCHS, 
                                optimizer=optimizer, 
                                early_stopping=None,
                                scheduler=scheduler,
                                cv_num=cv_num,
                                region=ROI,
                                model_load=MODEL_LOAD)

            trainer.load(cv_num, MODEL_LOAD_EPOCH) # pre-trained model load
            pred_ages, true_ages, features = trainer.test() # test

            pred_age_data.setdefault(v, [])
            true_age_data.setdefault(v, [])
            feature_data.setdefault(v, [])

            pred_age_data[v].extend(pred_ages)
            true_age_data[v].extend(true_ages)
            feature_data[v].extend(features)

        # save the data
        trainer.test_age_data_extraction(pred_age_data,
                                    true_age_data,
                                    feature_data)

        # pred_ages_filename = os.path.join(results_folder, 'pred_ages.pkl')
        # true_ages_filename = os.path.join(results_folder, 'true_ages.pkl')
        # features_filename = os.path.join(results_folder, 'features.pkl')
        # print(results_folder)

        # try:
        #     with open(pred_ages_filename, 'wb') as file:
        #         pickle.dump(pred_age_data, file)
        #     with open(true_ages_filename, 'wb') as file:
        #         pickle.dump(true_age_data, file)
        #     print(f"Ages saved")
        #     with open(features_filename, 'wb') as file:
        #         pickle.dump(feature_data, file)
        #     print(f"Features saved")
        # except Exception as e:
        #     print("Error saving data:", e)

   
    cv_num += 1






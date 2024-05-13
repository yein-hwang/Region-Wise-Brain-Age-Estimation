import os
import pandas as pd
import argparse

class Config:

    def __init__(self, args):
        
        for name, value in vars(args).items():
            setattr(self, name, value)
        
        dataset = {
            0: "ukb",
            1: "adni"
        }
        mri_csvs = {
            0: '/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/ukbb_cn_region.csv',
            1: '/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/adni_cn_region.csv' 
        }
        roots = {
            0: '/media/leelabsg-storage1/DATA/UKBB/bulk/20252_unzip', # UKBB
            1: '/media/leelabsg-storage1/yein/research/data/adni_region' # ADNI
        }
        self.old_root = roots[0]
        self.new_root = roots[1]
        self.mri_csv_old = mri_csvs[0]
        self.mri_csv_new = mri_csvs[1]
        self.input_size = (1, 128, 128, 128) 
        self.batch_size = args.batch_size
        self.num_cpu_workers = args.n_workers
        self.nb_epochs = args.epochs
        # Optimizer
        self.lr = args.lr # 
        self.weight_decay = 5e-5
        # early stoppping
        self.patience = args.patience
        # learning rate
        self.lr = args.lr
        # training
        self.regions = {0: 'imgs', 1: 'caudate', 2: 'cerebellum', 3: 'frontal_lobe', 4: 'insula', 5: 'occipital_lobe', 6: 'parietal_lobe', 7: 'putamen', 8: 'temporal_lobe', 9: 'thalamus'}
        # self.roi = self.regions[args.roi]
        self.proj_n = args.proj_n
        self.results_folder = os.path.join('../../test/ewc/', args.results_path)
        self.model_load_folder = os.path.join('../../model/region_BAE/ukb')
        self.model_save_folder = os.path.join('../../model/region_BAE/ewc/')
        self.model_load_epoch = args.load_epoch
        mode = {
            0: 'train',
            1: 'test',
            2: 'test_tf'
        }
        self.mode = mode[args.mode]
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--roi', type=int, default=0,
    #                     help='0: global, 1: caudate, 2: cerebellum, 3: frontal_lobe, 4: insula, 5: occipital_lobe, 6: parietal_lobe, 7: putamen, 8: temporal_lobe, 9: thalasmus')
    parser.add_argument('--proj_n', type=str, default='region-wise')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--mode', type=int, default=0, help='0: train, 1: test, 2: test_tf')
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--load_epoch', type=int, default=40)

    return parser.parse_args()
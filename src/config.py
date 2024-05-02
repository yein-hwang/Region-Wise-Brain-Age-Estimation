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
        roots = {
            0: '/media/leelabsg-storage1/DATA/UKBB/bulk/20252_unzip', # UKBB
            1: '/media/leelabsg-storage1/yein/research/data/adni_region' # ADNI
        }
        mri_csvs = {
            0: '/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/ukbb_cn_region.csv',
            1: '/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/adni_cn_region.csv' 
        }
        self.dataset = dataset[args.dataset]
        self.root = roots[args.dataset]
        self.mri_csv = mri_csvs[args.dataset]
        self.data_size = args.data_size
        self.input_size = (1, 128, 128, 128) 
        self.batch_size = args.batch_size
        self.pin_mem = True
        self.num_cpu_workers = args.n_workers
        self.cuda = True
        
        self.ensemble_number = args.ensemble_number
        self.nb_epochs = args.epochs
        self.lr = args.lr # Optimizer
        self.weight_decay = 5e-5
        self.patience = 1
        self.lr_scheduler_choice = args.lr_scheduler_choice
        self.regions = {0: 'imgs', 1: 'caudate', 2: 'cerebellum', 3: 'frontal_lobe', 4: 'insula', 5: 'occipital_lobe', 6: 'parietal_lobe', 7: 'putamen', 8: 'temporal_lobe', 9: 'thalamus'}
        self.roi = self.regions[args.roi]
        self.proj_n = args.proj_n
        self.results_folder = f'../../test/{self.dataset}'
        self.model_save_folder = f'../../model/region_BAE/ukb'
        self.model_load = args.model_load
        mode = {
            0: 'train',
            1: 'test',
            2: 'test_tf'
        }
        self.mode = mode[args.mode]
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_size', type=int, default=25657)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ensemble_number', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler_choice', type=int, default=1,
                        choices=[1, 2, 3], help='1: CustomCosineAnnealingWarmUpRestarts, 2: Cosine, 3: ReduceLROnPlateau')
    parser.add_argument('--roi', type=int, default=0,
                        help='0: global, 1: caudate, 2: cerebellum, 3: frontal_lobe, 4: insula, 5: occipital_lobe, 6: parietal_lobe, 7: putamen, 8: temporal_lobe, 9: thalasmus')
    parser.add_argument('--proj_n', type=str, default='region-wise')
    parser.add_argument('--model_load', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0: train, 1: test, 2: test_tf')

    return parser.parse_args()
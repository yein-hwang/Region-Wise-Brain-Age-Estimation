import pandas as pd
import os
import glob
from tqdm import tqdm
import ants
import gc
import argparse

class Config: 
    def __init__(self, args):
        for name, value in vars(args).items():
            setattr(self, name, value)
        
        self.trial = args.trial
        self.start_index = args.start_index
        self.end_index = args.end_index

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--trial', type=int, default=0)
        parser.add_argument('--start_index', type=int, default=0)
        parser.add_argument('--end_index', type=int, default=100)

        return parser.parse_args()

def process_subject(root, subject, atlas_template_path, save_root, regions, cn, trial, tmp_path):
    img_dir = os.path.join(root, subject + '_20252_2_0/T1/T1_brain_to_MNI.nii.gz')
    save_path = os.path.join(save_root, subject)

    normal_case = glob.glob('/media/leelabsg-storage1/yein/research/data/ukbb_3/1363458/*')
    curr_check = glob.glob(save_path + '/*')

    if not len(curr_check) == len(normal_case):
        os.makedirs(save_path, exist_ok=True)

        image = ants.image_read(img_dir)
        image = ants.resample_image(image, (128, 128, 128), 1, 0)
        mask = ants.get_mask(image)

        img_path = save_path + '/T1w_registered.nii.gz'
        mask_path = save_path + '/T1w_brain_mask_registered.nii.gz'

        ants.image_write(image, img_path)
        ants.image_write(mask, mask_path)

        transformation = ants.registration(
            fixed=image,
            moving=ants.image_read(atlas_template_path), 
            type_of_transform='SyN',
            outprefix=tmp_path
        )
        registered_atlas_ants = transformation['warpedmovout']
        gc.collect()

        for region_idx in range(1, 10):
            region_mask = registered_atlas_ants == region_idx
            region_mask_dilated = ants.morphology(region_mask, radius=4, operation='dilate', mtype='binary')

            extracted_region = image.numpy() * region_mask_dilated.numpy()
            extracted_region_ants = ants.from_numpy(extracted_region)

            region_path = save_path + f'/region_{region_idx}.nii.gz'
            region_mask_path = save_path + f'/region_{region_idx}_mask.nii.gz'
            ants.image_write(extracted_region_ants, region_path)
            ants.image_write(region_mask_dilated, region_mask_path)

            del region_mask, region_mask_dilated, extracted_region
            gc.collect()
        

def main(config):
    gc.collect()

    root = '/media/leelabsg-storage1/DATA/UKBB/bulk/20252_uz/'
    cn = pd.read_csv('/media/leelabsg-storage1/yein/research/data/csv/ukbb_cn.csv')

    data_size = 10000
    trial = config.trial
    save_root = f'/media/leelabsg-storage1/yein/research/data/ukbb_region_{data_size}'
    tmp_path = '/media/leelabsg-storage1/yein/research/tmp/tmp'
    os.makedirs(save_root, exist_ok=True)

    atlas_template_path = '/media/leelabsg-storage1/yein/research/data/template/MNI-maxprob-thr0-1mm.nii.gz'
    regions = ['caudate', 'cerebellum', 'frontal_lobe', 'insula', 'occipital_lobe', 'parietal_lobe', 'putamen', 'temporal_lobe', 'thalamus']

    for i in tqdm(cn.index[config.start_index : config.end_index]):
        subject = str(cn['id'][i])
        process_subject(root, subject, atlas_template_path, save_root, regions, cn, trial, tmp_path)
        gc.collect()

if __name__ == "__main__":
    args = Config.parse_args()
    config = Config(args)
    main(config)
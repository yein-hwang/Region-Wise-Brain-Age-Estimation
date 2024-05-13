import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import nibabel as nib
from skimage.transform import resize

class UKB_Dataset(Dataset):
    def __init__(self, config, indices=None):
        super(UKB_Dataset, self).__init__()
        self.config = config
        self.data_dir = config.data
        self.data_csv = pd.read_csv(config.label)
        
        self.image_names = [self.data_csv['label'][i] for i in indices]
        self.labels = [self.data_csv['age'][i] for i in indices]
        
        print("images_names[100]: ", self.image_names[100], ", images_names[-1]: ", self.image_names[-1])
        print("labels[100]:       ", self.labels[100], ",      labels[-1]:       ", self.labels[-1])
    
        self.transform = T.Compose([
            T.ToTensor()
        ])
    def collate_fn(self, batch):
        images, labels = zip(*batch)  # separate images and labels
        images = torch.stack(images)  # stack images into a tensor
        labels = torch.tensor(labels)  # convert labels into a tensor
        return images, labels

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]

        # Load the image
        image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.float32)
        image = torch.from_numpy(image).float()
        image = image.permute(3, 0, 1, 2)

        np.random.seed()
        age = torch.tensor(label, dtype=torch.float32)

        return (image, age)
    

class ADNI_Dataset(Dataset):
    def __init__(self, image_path, label_path, indices, image_size=128):
        super(ADNI_Dataset, self).__init__()
        self.data_dir = image_path
        self.data_csv = pd.read_csv(label_path)
        
        # Use 'file_name' column to create image paths
        self.files = self.data_csv.loc[indices, 'File_name'].tolist()
        print(f"Number of files loaded: {len(self.files)}")  # Add this line

        self.input_size = (1, 128, 128, 128)

    def __getitem__(self, index):
        label = self.data_csv.loc[index, 'age']
        file_name = self.data_csv.loc[index, 'File_name']

        # Use 'file_name' to construct the image path
        image_path = os.path.join(self.data_dir, file_name, 'brain_to_MNI_nonlin.nii.gz')

        # Convert Nifti1Image to numpy array
        image = nib.load(image_path).get_fdata()

        # Resize the image to desired size if necessary
        image = resize(image, self.input_size)

        # Convert numpy array to PyTorch tensor
        image = torch.from_numpy(image).float()

        age = torch.tensor(label, dtype=torch.float32)
        
        return (image, age)
    
    def __len__(self):
        return len(self.files)

    def collate_fn(self, batch):
        images, labels = zip(*batch)  # separate images and labels
        images = torch.stack(images)  # stack images into a tensor
        labels = torch.tensor(labels)  # convert labels into a tensor
        return images, labels


class Region_Dataset(Dataset):
    def __init__(self, root, mri_csv, indices=None, roi=None):
        super(Region_Dataset, self).__init__()
        self.data_dir = root
        self.data_csv = mri_csv
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
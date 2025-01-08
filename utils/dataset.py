import os
import numpy as np
import torch
from skimage.io import imread
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, split_file_path, transform=None):
        self.root_dir = root_dir
        with open(split_file_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        self.transform = transform
        self.class_names = sorted(os.listdir(os.path.join(root_dir, 'EuroSAT_RGB')))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class EuroSATMS_Dataset(Dataset):
    def __init__(self, root_dir, split_file_path, transform=None):        
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(os.path.join(root_dir, 'EuroSAT_MS')))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        with open(split_file_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = imread(img_path)    
        image = image.astype(np.float32) / 65535.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
    
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[label_name]

        if self.transform is not None:

            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip, transforms.RandomAffine)):
                        image = t(image)

            rgb_channels = image[1:4]
            transformed_rgb = rgb_channels

            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.ColorJitter):
                        transformed_rgb = t(transformed_rgb)

            final_tensor = torch.zeros_like(image)
            final_tensor[:1] = image[:1]
            final_tensor[4:] = image[4:]
            final_tensor[1:4] = transformed_rgb

            return final_tensor, label

        return image, label

    def get_class_names(self):
        return self.class_names
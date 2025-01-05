import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
import matplotlib.pyplot as plt

class EuroSATMS_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):        
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.samples = []

        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            class_to_idx = self.class_to_idx[class_name]
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.tif'):
                    img_path = os.path.join(class_dir, file_name)
                    self.samples.append((img_path, class_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = imread(img_path)
            image = image.astype(np.float32) / 65535.0

            image = torch.from_numpy(image.transpose(2, 0, 1))

            if self.transform is not None:
                image = self.transform(image)
            
            return image, label

        def get_class_names(self):
            return self.class_names

def load_and_preprocess(image_path):
    image = imread(image_path)
    image = image.astype(np.float32) / 65535.0
    image = torch.from_numpy(image.transpose(2,0,1))
    plt.figure(figsize= (10,5))
    return image

if __name__ == "__main__":
    sample_path = r'D:\EuroSAT_MS\EuroSAT_MS\AnnualCrop\AnnualCrop_1.tif'
    single_image = load_and_preprocess(sample_path)
    print(single_image.shape)
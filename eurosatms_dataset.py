import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
import matplotlib.pyplot as plt



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
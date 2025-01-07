import os
import torch
import numpy as np
from skimage.io import imread
from models.resnet18_model import ResNet18
from models.mynet import FeatureExtractor, MyNet
from utils.transforms import get_transforms
from utils.dataset import EuroSATMS_Dataset
from torch.utils.data import DataLoader
from scripts.train import train_model
from utils.utils import plot_tpr, plot_validation

def load_and_preprocess(image_path):
    image = imread(image_path)
    image = image.astype(np.float32) / 65535.0
    image = torch.from_numpy(image.transpose(2,0,1))
    return image

def main():
    dataset_root = r'D:\EuroSAT_MS'
    split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    transform_v1, transform_v2 = get_transforms()

    datasets = []

    for transform in [transform_v1, transform_v2]:
        train_dataset = EuroSATMS_Dataset(dataset_root, transform=transform['train'], split_file_path = os.path.join(split_dir, 'train.txt'))
        val_dataset =  EuroSATMS_Dataset(dataset_root, transform=None, split_file_path = os.path.join(split_dir, 'val.txt'))

        datasets.append((train_dataset, val_dataset))

    results = []

    for i, (train, val) in enumerate(datasets):

        train_loader = DataLoader(train, batch_size = 32, shuffle = True)
        val_loader = DataLoader(val, batch_size = 32, shuffle = False)

        resnet = ResNet18(num_classes=10).get_model()
        model = MyNet(num_classes=10, pretrained_model1=resnet)
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        val_acc, class_tpr = train_model(model, train_loader, val_loader,
                                        device, criterion, optimizer)

        results.append((val_acc, class_tpr))

    plot_validation(results=results)
    plot_tpr(results=results)

if __name__ == "__main__":
    main()
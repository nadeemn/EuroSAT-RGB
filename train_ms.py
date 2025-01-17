import os
import argparse
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

def main(root_dir):
    dataset_root = root_dir
    split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_v1, transform_v2 = get_transforms()

    datasets = []

    for transform in [transform_v1, transform_v2]:
        train_dataset = EuroSATMS_Dataset(dataset_root, transform=transform['train'], split_file_path = os.path.join(split_dir, 'train.txt'))
        val_dataset =  EuroSATMS_Dataset(dataset_root, transform=None, split_file_path = os.path.join(split_dir, 'val.txt'))

        datasets.append((train_dataset, val_dataset))

    results = []
    best_accuracy = 0.0

    for i, (train, val) in enumerate(datasets):

        train_loader = DataLoader(train, batch_size = 32, shuffle = True)
        val_loader = DataLoader(val, batch_size = 32, shuffle = False)

        resnet = ResNet18(num_classes=10).get_model()
        model = MyNet(num_classes=10, pretrained_model1=resnet)
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        val_acc, class_tpr, best_accuracy = train_model(model, train_loader, val_loader,
                                        device, criterion, optimizer, best_accuracy, best_model = 'best_model_ms')

        results.append((val_acc, class_tpr))

    plot_validation(results=results)
    plot_tpr(results=results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('--root_dir', type=str, required=True, help="""Root Directory of the dataset. 
                        (No need to give the entire directory. Only parent directory is enough.)
                        For e.g. if the file path is: D:\EuroSAT_MS\EuroSAT_MS\AnnualCrop\AnnualCrop_1.tif.
                        Give the root as D:\EuroSAT_MS
                        """ )
    args = parser.parse_args()
    main(root_dir = args.root_dir)
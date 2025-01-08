import torch
import argparse
import os
from torch.utils.data import DataLoader
from utils.transforms import get_transforms
from utils.dataset import EuroSATDataset
from models.resnet18_model import ResNet18
from scripts.train import train_model
from utils.utils import plot_tpr, plot_validation

def main(root_dir):
    dataset_root = root_dir
    splits_dir = './splits'
    torch.manual_seed(29122024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_v1, transform_v2 = get_transforms()

    datasets = []

    for transform in [transform_v1, transform_v2]:
        train_dataset = EuroSATDataset(dataset_root, transform=transform['train'], split_file_path = os.path.join(splits_dir, 'train.txt'))
        val_dataset =  EuroSATDataset(dataset_root, transform=transform['val'], split_file_path = os.path.join(splits_dir, 'val.txt'))

        datasets.append((train_dataset, val_dataset))

    results = []
    best_accuracy = 0.0

    for i, (train, val) in enumerate(datasets):

        train_loader = DataLoader(train, batch_size = 32, shuffle = True)
        val_loader = DataLoader(val, batch_size = 32, shuffle = False)

        resnet_model = ResNet18(len(train.class_names))
        model = resnet_model.get_model()
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        val_acc, class_tpr, best_accuracy = train_model(model, train_loader, val_loader,
                                        device, criterion, optimizer, best_accuracy)

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
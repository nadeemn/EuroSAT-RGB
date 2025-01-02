import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

dataset_root = r'D:\EuroSAT_RGB'
splits_dir = './splits'

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

def get_transforms():
    transform_1 = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ])
    }

    transform_2 = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9,1.1)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Compose([
                transforms.ToTensor()
            ])
        ])
    } 

    return transform_1, transform_2

def train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=10):
    best_accuracy = 0.0
    validation_accuracies = []
    class_tpr_over_epochs = {class_name: [] for class_name in train_loader.dataset.class_names}

    for epoch in range(10):
        # Training phase
        model.train()
        correct, total = 0,0
        running_loss = 0.0

        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # validation phase
        model.eval()
        val_preds, val_labels = [], []
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
             for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        validation_accuracies.append(val_acc)

        # save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': best_accuracy
            }, 'best_model.pth')

        report = classification_report(val_labels, val_preds, target_names = val_loader.dataset.class_names, output_dict = True)

        for class_name in class_tpr_over_epochs:
            class_tpr_over_epochs[class_name].append(report[class_name]['recall'])

        print(f'Epoch {epoch + 1}: Train Loss={running_loss/len(train_loader):.4f},'
                f'Train Acc = {100 * correct / total:.2f}%, Val Acc = {val_acc:.2f}%')

    return validation_accuracies, class_tpr_over_epochs


if __name__ == "__main__":

    torch.manual_seed(29122024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_v1, transform_v2 = get_transforms()

    datasets = []

    for transform in [transform_v1, transform_v2]:
        train_dataset = EuroSATDataset(dataset_root, transform=transform['train'], split_file_path = os.path.join(splits_dir, 'train.txt'))
        val_dataset =  EuroSATDataset(dataset_root, transform=transform['val'], split_file_path = os.path.join(splits_dir, 'val.txt'))

        datasets.append((train_dataset, val_dataset))

    results = []

    for i, (train, val) in enumerate(datasets):

        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

        model = resnet18(weights = ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.class_names))
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        val_acc, class_tpr = train_model(model, train_loader, val_loader,
                                        device, criterion, optimizer)

        results.append((val_acc, class_tpr))

    plt.figure(figsize=(15, 5))
    for i, (val_acc, _) in enumerate(results):
        plt.plot(range(1, len(val_acc) + 1), val_acc, label= f'Augmentation: {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy %')
    plt.title('Validation accuracy over epochs')
    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 6))
    for i, (_, class_tpr) in enumerate(results):
        for class_name, tpr in class_tpr.items():
            plt.plot(range(1, len(tpr) + 1), tpr, label=f'{class_name} Augmentation: {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('TPR')
    plt.title('TRP per class over epochs')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

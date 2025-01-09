import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision.utils import make_grid
from models.resnet18_model import ResNet18
from models.mynet import MyNet

def load_model(model_path, num_classes, device):
    """Load the trained model"""
    model = ResNet18(num_classes).get_model()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

def load_mynet_model(model_path, num_classes, device):
    """Load the trained MYNet Model"""
    resnet = ResNet18(num_classes).get_model()
    mynet = MyNet(num_classes=num_classes, pretrained_model1=resnet)
    checkpoint = torch.load(model_path, weights_only=True)
    mynet.load_state_dict(checkpoint['model_state_dict'])
    model = mynet.to(device)
    return model

def save_and_load_logits(logits, logits_path, save_mode = True):
    """ Save or compare logits for reproduction testing"""

    if save_mode:
        torch.save(logits, logits_path)
        print(f"Logits save to {logits_path}")
    else:
        if os.path.exists(logits_path):
            saved_logits = torch.load(logits_path, weights_only=True)
            max_diff = torch.max(torch.abs(logits - saved_logits))
            print(f"Max difference between saved and current logits: {max_diff:.6f}")
            return max_diff
        else:
            print("no saved logits found")
            return None
        
def plot_top_bottom_images(dataset, class_scores, class_names, num_classes=3, save_dir='results'):
    """Plot and save top/bottom images for random 3 classes"""
    os.makedirs(save_dir, exist_ok=True)
    selected_classes = list(class_names)[:num_classes]

    for class_name in selected_classes:
        scores = class_scores[class_name]
        sorted_scores = sorted(scores, key= lambda x: x['score'], reverse=True)
        top_5 = sorted_scores[:5]
        bottom_5 = sorted_scores[-5:]

        for category, images in [("top", top_5), ("bottom", bottom_5)]:
            category_images = [dataset[s["image_id"]][0]for s in images]
            grid = make_grid(category_images, nrow=5, normalize=True, pad_value=0.5)

            plt.figure(figsize=(15, 5))
            plt.title(f"{category.capitalize()} 5 scoring images for class {class_name}")
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"{class_name}_{category}.png"))
            plt.close()

def save_test_results(report, save_dir="results"):
    """Save the classification report and metrics"""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "test_results.txt"), 'w') as f:
        f.write(f"Test Results \n")
        f.write("=" * 50 + "\n\n")
        json.dump(report, f, indent=4)

def plot_validation(results):
    """Plot Validation Accuracy"""
    plt.figure(figsize=(15, 5))
    for i, (val_acc, _) in enumerate(results):
        plt.plot(range(1, len(val_acc) + 1), val_acc, label= f'Augmentation: {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy %')
    plt.title('Validation accuracy over epochs')
    plt.legend()
    plt.show()

def plot_tpr(results):
    """Plot TPR per class over the epochs"""
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

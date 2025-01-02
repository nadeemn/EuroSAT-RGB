import torch
import os
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from train_model import EuroSATDataset
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import classification_report


def load_model(model_path, num_classes, device):
    """Load the trained model"""
    model = resnet18(weights = None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

def save_and_load_logits(logits, save_mode = True, logits_path='test_logits.pt'):
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

def evaluate_model(model, test_loader, device, class_names, save_logits=False):
    """Evaluate model on the test set and return prediction, labels and logits."""

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    class_scores = {class_name: [] for class_name in class_names}

    with torch.no_grad():
        for batch , (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            probabilites = torch.softmax(outputs, dim =1)
            _, predicted_class = torch.max(outputs, 1)

            all_preds.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i, (label, probs) in enumerate(zip(labels, probabilites)):
                class_name = class_names[label]
                image_idx = batch * test_loader.batch_size + i
                class_scores[class_name].append({
                    "score": probs[label].item(),
                    "probabilites": probs.cpu().numpy(),
                    "image_id": image_idx
                })
    
    all_logits = torch.cat(all_logits, dim = 0)
    save_and_load_logits(all_logits, save_mode = save_logits)

    return all_preds, all_labels, class_scores

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

if __name__ == "__main__":

    dataset_root = r'D:\EuroSAT_RGB'
    splits_dir = './splits'
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = EuroSATDataset(dataset_root, transform=test_transform, split_file_path = os.path.join(splits_dir, 'test.txt'))
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    class_names = test_dataset.class_names

    model = load_model('best_model.pth', len(class_names), device)

    all_preds, all_labels, class_scores = evaluate_model(model, test_loader, device, class_names, save_logits=False)

    report = classification_report(all_labels, all_preds, target_names = class_names, output_dict=True)

    save_test_results(report, results_dir)
    plot_top_bottom_images(test_dataset, class_scores, class_names, num_classes=3, save_dir=results_dir)

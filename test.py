import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from task2 import EuroSATDataset
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import classification_report

dataset_root = r'D:\EuroSAT_RGB'
splits_dir = './splits'
class_names = sorted(os.listdir(os.path.join(dataset_root, 'EuroSAT_RGB')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_scores = {class_name: [] for class_name in class_names}

test_transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = EuroSATDataset(dataset_root, transform=test_transform, split_file_path = os.path.join(splits_dir, 'test.txt'))
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

model = resnet18(pretrained = True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []


with torch.no_grad():
    for batch , (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilites = torch.nn.functional.softmax(outputs, dim =1).squeeze().cpu().numpy()
        _, predicted_class = torch.max(outputs, 1)

        all_preds.extend(predicted_class.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(len(labels)):
            label = labels[i].item()
            class_name = class_names[label]

            class_scores[class_name].append({
                "score": probabilites[i][label],
                "probabilites": probabilites[i],
                "image_id": batch * test_loader.batch_size + i
            })

report = classification_report(all_labels, all_preds, target_names = class_names)
#print(report)

top_bottom_images = {}

for class_name, scores in class_scores.items():
    sorted_scores = sorted(scores, key = lambda x: x['score'], reverse= True)
    top_5 = sorted_scores[:5]
    bottom_5 = sorted_scores[-5:]
    top_bottom_images[class_name] = {"top" : top_5, "bottom": bottom_5}

selected_classes = list(top_bottom_images.keys())[:3]  # Pick 3 classes

for class_name in selected_classes:
    print(f"Class: {class_name}")
    for category in ["top", "bottom"]:
        category_images = [test_dataset[s["image_id"]][0] for s in top_bottom_images[class_name][category]]
        grid = make_grid(category_images, nrow=5, normalize=True, pad_value=0.5)
        plt.figure(figsize=(15, 5))
        plt.title(f"{category.capitalize()} 5 scoring images for class {class_name}")
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
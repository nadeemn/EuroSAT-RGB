import torch
import os
from torch.utils.data import DataLoader
from utils.dataset import EuroSATDataset
from scripts.test import evaluate_model
from torchvision import transforms
from utils.utils import load_model, save_test_results, plot_top_bottom_images
from sklearn.metrics import classification_report

def test():
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

if __name__ == "__main__":
    test()
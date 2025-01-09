import torch
import argparse
import os
from torch.utils.data import DataLoader
from utils.dataset import EuroSATDataset
from scripts.test import evaluate_model
from torchvision import transforms
from utils.utils import load_model, save_test_results, plot_top_bottom_images
from sklearn.metrics import classification_report

def test(root_dir, save_logits=False):
    dataset_root = root_dir
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

    model = load_model('best_model_rgb.pth', len(class_names), device)

    all_preds, all_labels, class_scores = evaluate_model(model, test_loader, device, class_names, logits_path="test_logits_rgb.pt", save_logits=save_logits)

    report = classification_report(all_labels, all_preds, target_names = class_names, output_dict=True)

    save_test_results(report, results_dir)
    plot_top_bottom_images(test_dataset, class_scores, class_names, num_classes=3, save_dir=results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing")
    parser.add_argument('--root_dir', type=str, required=True, help="""Root Directory of the dataset. 
                        (No need to give the entire directory. Only parent directory is enough.)
                        For e.g. if the file path is: D:\EuroSAT_MS\EuroSAT_MS\AnnualCrop\AnnualCrop_1.tif.
                        Give the root as D:\EuroSAT_MS
                        """ )
    parser.add_argument('--save_logits', type=bool, required=False, help="If you want to save the logits. Give i/p as true")
    args = parser.parse_args()
    test(root_dir=args.root_dir, save_logits = args.save_logits)
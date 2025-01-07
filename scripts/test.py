import torch
from utils.utils import save_and_load_logits

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


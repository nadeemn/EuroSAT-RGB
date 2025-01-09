import torch
from sklearn.metrics import classification_report

def train_model(model, train_loader, val_loader, device, criterion, optimizer, best_accuracy, best_model, num_epochs=10):
    best_accuracy = best_accuracy
    validation_accuracies = []
    class_tpr_over_epochs = {class_name: [] for class_name in train_loader.dataset.class_names}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct, total = 0,0
        running_loss = 0.0

        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
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
            }, f'{best_model}.pth')

        report = classification_report(val_labels, val_preds, target_names = val_loader.dataset.class_names, output_dict = True, zero_division=0)

        for class_name in class_tpr_over_epochs:
            class_tpr_over_epochs[class_name].append(report[class_name]['recall'])

        print(f'Epoch {epoch + 1}: Train Loss={running_loss/len(train_loader):.4f},'
                f'Train Acc = {100 * correct / total:.2f}%, Val Acc = {val_acc:.2f}%')

    return validation_accuracies, class_tpr_over_epochs, best_accuracy

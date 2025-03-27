import torch
from tqdm import tqdm

def create_confusion_matrix(model, test_loader, device, num_classes=12):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Creating confusion matrix"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    return confusion_matrix
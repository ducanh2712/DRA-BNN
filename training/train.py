import torch
from tqdm import tqdm

def train_FSG(model, train_loader, optimizer, criterion, device, dvlr, alpha_list):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for param in model.parameters():
            if hasattr(param, 'alpha'):
                param.grad *= 0.5
                grad_alpha = param.grad.sum().item()
                param.alpha.data = dvlr.update(grad_alpha)
                alpha_list.append(param.alpha.item())

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total
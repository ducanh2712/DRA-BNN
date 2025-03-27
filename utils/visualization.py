import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.float() / cm.sum(axis=1)[:, None]
    cm_normalized = cm_normalized.cpu().numpy()
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

def visualize_samples(dataloader, class_names, num_samples=12, save_path=None):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sample images saved to {save_path}")
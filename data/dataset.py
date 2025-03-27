import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class GarbageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_names = class_dirs
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(data_dir, class_name)
            image_files = glob.glob(os.path.join(class_path, '*.jpg')) + \
                         glob.glob(os.path.join(class_path, '*.jpeg')) + \
                         glob.glob(os.path.join(class_path, '*.png'))
            
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                
        print(f"Loaded {len(self.image_paths)} images from {len(class_dirs)} classes")
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if self.transform:
                placeholder = torch.zeros((3, 224, 224))
                return placeholder, label
            else:
                placeholder = Image.new('RGB', (224, 224), color='black')
                return placeholder, label
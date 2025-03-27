import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data import GarbageDataset, get_train_transform, get_val_transform
from models import BinaryResNet18
from training import train_FSG, validate, DVLR
from utils import calculate_flops, create_confusion_matrix, plot_confusion_matrix, visualize_samples, parse_config
import matplotlib.pyplot as plt
import argparse
import random

# Constants for testing
TEST_MODEL = "/home/njlab/Desktop/Anh_san/project_root/epochs_97_82.86_log/best_model.pth"
NUM_TEST_IMAGES = 200  # Number of images to test
NUM_DISPLAY_IMAGES = 20  # Number of images to display as examples

def create_datasets(data_dir, batch_size):
    """Creates and returns train/val datasets and dataloaders."""
    temp_dataset = GarbageDataset(data_dir)
    dataset_size = len(temp_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = torch.utils.data.Subset(
        GarbageDataset(data_dir, get_train_transform()),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        GarbageDataset(data_dir, get_val_transform()),
        val_indices
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, temp_dataset.class_names

def plot_individual_graph(data_list, label, filename, title, highlight_value=None, highlight_label=None):
    """Plots a single graph and saves it to a file."""
    plt.figure(figsize=(10, 6))
    plt.plot(data_list, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    
    if highlight_value is not None and highlight_label is not None:
        plt.axhline(y=highlight_value, color='r', linestyle='--', alpha=0.5)
        plt.text(len(data_list)//2, highlight_value, f'{highlight_label}: {highlight_value:.2f}', 
                 color='red', ha='center', va='bottom')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()
def test_model(model, dataset, class_names, device, test_dir, num_test_images=NUM_TEST_IMAGES, num_display_images=9):
    """Test the model on a specified number of images and save results."""
    model.eval()
    inference_times = []
    predictions = []
    true_labels = []
    images_to_show = []

    # Warm-up the model to reduce initial latency
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):  # Run a few dummy inferences
            model(dummy_img)

    # Create test result directory
    os.makedirs(test_dir, exist_ok=True)

    # Randomly select images
    indices = random.sample(range(len(dataset)), num_test_images)
    
    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Measure inference time
            start_time = time.time()
            output = model(img)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            _, pred = torch.max(output, 1)
            predictions.append(pred.item())
            true_labels.append(label)
            if len(images_to_show) < num_display_images:
                images_to_show.append((img.cpu(), pred.item(), label))

    avg_inference_time = np.mean(inference_times)
    accuracy = 100 * sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
    
    # Visualize 9 predictions in a 3x3 grid
    plt.figure(figsize=(12, 12))  # Điều chỉnh kích thước tổng thể của hình
    for i, (img, pred, true) in enumerate(images_to_show[:9]):  # Chỉ lấy 9 ảnh
        ax = plt.subplot(3, 3, i + 1)  # Lưới 3x3
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"Pred: {class_names[pred]}\nTrue: {class_names[true]}", 
                  fontsize=14,  # Tăng kích thước chữ
                  fontweight='bold')  # Làm chữ đậm
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, "test_predictions.png"))
    print(f"Test predictions (9 examples) saved to {test_dir}/test_predictions.png")
    
    # Save results
    with open(os.path.join(test_dir, "test_results.txt"), "w") as f:
        f.write(f"Tested {num_test_images} random images\n")
        f.write(f"Average Inference Time: {avg_inference_time:.2f} ms\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("Individual Inference Times:\n")
        for i, t in enumerate(inference_times):
            f.write(f"Image {i+1}: {t:.2f} ms\n")
        f.write("\nPredictions:\n")
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            f.write(f"Image {i+1}: Predicted = {class_names[pred]}, True = {class_names[true]}\n")
    
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test results saved to {test_dir}/test_results.txt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train or Test BinaryResNet18")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    # Parse configuration from XML
    config = parse_config('config.xml')
    
    # Base log directory
    base_log_dir = "./log_result"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Specific directories for train and test
    train_dir = os.path.join(base_log_dir, "train_result")
    test_dir = os.path.join(base_log_dir, "test_result")
    config['log_dir'] = train_dir  # Update config log_dir for training
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else config['device'])

    if args.mode == 'train':
        # Create train directory
        os.makedirs(train_dir, exist_ok=True)

        # Setup datasets and dataloaders
        train_loader, val_loader, class_names = create_datasets(config['data_dir'], config['batch_size'])

        # Model and training setup
        model = BinaryResNet18(num_classes=config['num_classes']).to(device)
        dvlr = DVLR(lr=0.01)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=config['min_lr'], verbose=True
        )

        # Logging and visualization
        flops, params, flops_str, params_str = calculate_flops(model)
        with open(os.path.join(train_dir, "model_complexity.txt"), "w") as f:
            f.write(f"Model: BinaryResNet18\nFLOPs: {flops_str}\nParameters: {params_str}\nClasses: {config['num_classes']}\n")
        print(f"Model FLOPs: {flops_str}\nModel Parameters: {params_str}")
        visualize_samples(train_loader, class_names, save_path=os.path.join(train_dir, "sample_images.png"))

        # Training loop
        train_loss_list, val_acc_list, alpha_list, epoch_time_list = [], [], [], []
        best_acc, best_epoch = 0.0, 0
        early_stop_counter, min_loss = 0, float('inf')
        log_file = open(os.path.join(train_dir, "training_log.txt"), "w")
        log_file.write(f"Dataset: Garbage Classification\nClasses: {', '.join(class_names)}\n\nTraining Log:\n" + "-"*80 + "\n")

        try:
            for epoch in range(config['num_epochs']):
                start_time = time.time()
                avg_loss = train_FSG(model, train_loader, optimizer, criterion, device, dvlr, alpha_list)
                acc = validate(model, val_loader, device)
                scheduler.step(avg_loss)
                epoch_time = time.time() - start_time
                
                train_loss_list.append(avg_loss)
                val_acc_list.append(acc)
                epoch_time_list.append(epoch_time)
                
                log_entry = f'Epoch {epoch+1}: Time = {epoch_time:.2f}s, Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%, LR = {optimizer.param_groups[0]["lr"]:.6f}'
                print(log_entry)
                log_file.write(log_entry + "\n")
                
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(train_dir, "best_model.pth"))
                    log_file.write(f"New best accuracy: {best_acc:.2f}% at epoch {epoch+1}\n")
                    log_file.flush()  # Ensure best model log is written immediately
                
                if avg_loss < min_loss - config['early_stopping_delta']:
                    min_loss = avg_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    log_file.write(f"Early stopping triggered at epoch {epoch+1}\n")
                    break

        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at epoch {len(train_loss_list) + 1}")
            log_file.write(f"\nTraining interrupted by user at epoch {len(train_loss_list) + 1}\n")
            
            # Save the current model state
            current_model_path = os.path.join(train_dir, f"current_model_epoch_{len(train_loss_list)}.pth")
            torch.save(model.state_dict(), current_model_path)
            print(f"Current model saved at: {current_model_path}")
            log_file.write(f"Current model saved at: {current_model_path}\n")

        finally:
            # Calculate averages
            avg_epoch_time = sum(epoch_time_list) / len(epoch_time_list) if epoch_time_list else 0.0
            avg_train_loss = sum(train_loss_list) / len(train_loss_list) if train_loss_list else 0.0
            max_acc = max(val_acc_list) if val_acc_list else 0.0
            
            # Summary
            summary = (
                f"\nTraining Summary:\n"
                f"- Total Epochs: {len(epoch_time_list)}\n"
                f"- Average Time per Epoch: {avg_epoch_time:.2f}s\n"
                f"- Average Training Loss: {avg_train_loss:.4f}\n"
                f"- Best Accuracy: {best_acc:.2f}% at epoch {best_epoch}\n"
            )
            print(summary)
            log_file.write(summary)

            # Close log file
            log_file.close()

            # Generate and save confusion matrix
            confusion_matrix = create_confusion_matrix(model, val_loader, device, config['num_classes'])
            plot_confusion_matrix(confusion_matrix, class_names, os.path.join(train_dir, "confusion_matrix.png"))
            
            # Plot and save individual graphs
            plot_individual_graph(
                val_acc_list, 'Validation Accuracy (%)', os.path.join(train_dir, "accuracy_plot.png"),
                'Validation Accuracy Over Time', max_acc, 'Max Accuracy'
            )
            plot_individual_graph(
                epoch_time_list, 'Time (seconds)', os.path.join(train_dir, "time_plot.png"),
                'Training Time per Epoch', avg_epoch_time, 'Avg Time'
            )
            plot_individual_graph(
                train_loss_list, 'Training Loss', os.path.join(train_dir, "loss_plot.png"),
                'Training Loss Over Time', avg_train_loss, 'Avg Loss'
            )
            
            # Rename train directory with final results
            new_train_dir = os.path.join(base_log_dir, f"train_epochs_{len(train_loss_list)}_{best_acc:.2f}_log")
            os.rename(train_dir, new_train_dir)
            print(f"Training completed! Results saved in {new_train_dir}")
            print(f"Best model saved at: {os.path.join(new_train_dir, 'best_model.pth')}")

    elif args.mode == 'test':
        # Load the best model
        best_model_path = TEST_MODEL
        if not os.path.exists(best_model_path):
            print(f"Error: Best model not found at {best_model_path}. Please train the model first.")
            exit(1)

        # Initialize model and load state
        model = BinaryResNet18(num_classes=config['num_classes']).to(device)
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

        # Create full dataset for testing
        test_dataset = GarbageDataset(config['data_dir'], get_val_transform())
        class_names = test_dataset.class_names

        # Run test and save to test_dir
        test_model(model, test_dataset, class_names, device, test_dir)

if __name__ == "__main__":
    main()
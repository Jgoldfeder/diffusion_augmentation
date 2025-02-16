import torchvision.transforms as transforms
import argparse

from old.CustomDataset import FewShotDataset, ClassicalDataset

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch import nn
import torch
import wandb
import os
import random
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Few-shot learning training script')
    parser.add_argument('--subset', type=int, default=41, help='Random subset')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset name')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways')
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    wandb.init(
        project="classical-augmentation-experiments",
        config={
            "subset": args.subset,
            "shots": args.shots,
            "dataset": args.dataset,
            "ways": args.ways,
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": "resnet50",
        }
    )
    
    num_iterations = 10
    final_test_accuracies = []  # Add this list to store final accuracies
    for i in range(num_iterations):
        random.seed(time.time())

        base_path = f'few_shot_datasets/{args.dataset}/{args.ways}_ways/{args.shots}_shot/subset_{args.subset}'
        train_dataset = FewShotDataset(base_path, dataset_type='train')
        test_dataset = FewShotDataset(base_path, dataset_type='test')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        classical_dataset = ClassicalDataset(train_dataset, transform, duplicate_factor=6)

        print(len(train_dataset))
        print(train_dataset[0])

        train_loader = DataLoader(classical_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        #train the model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, args.ways)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(400):
            model.train()
            epoch_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels, class_names in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels, class_names in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            train_accuracy = 100 * train_correct / train_total
            test_accuracy = 100 * test_correct / test_total
            avg_loss = epoch_loss / len(train_loader)
            
            print(f'Epoch {epoch+1}/{400}, '
                f'Loss: {avg_loss:.4f}, '
                f'Train Accuracy: {train_accuracy:.2f}%, '
                f'Test Accuracy: {test_accuracy:.2f}%')

            if epoch == 399:  # Last epoch
                final_test_accuracies.append(test_accuracy)
                print(f"Final test accuracy for iteration {i+1}: {test_accuracy:.2f}%")

            wandb.log({
                "iteration": i + 1,
                "epoch": epoch + 1,
                "loss": avg_loss,
                f"train_accuracy_iteration_{i+1}": train_accuracy,
                f"test_accuracy_iteration_{i+1}": test_accuracy
            })
            
    # Calculate and log the average test accuracy
    avg_final_test_accuracy = sum(final_test_accuracies) / len(final_test_accuracies)
    print(f"\nAverage final test accuracy across all iterations: {avg_final_test_accuracy:.2f}%")
    wandb.log({"average_final_test_accuracy": avg_final_test_accuracy})

    wandb.finish()
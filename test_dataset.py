import torchvision.transforms as transforms
import argparse

from CustomDataset import FewShotDataset, ClassicalDataset

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch import nn
import torch
import wandb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Few-shot learning training script')
    parser.add_argument('--seed', type=int, default=41, help='Random seed')
    parser.add_argument('--shots', type=int, default=2, help='Number of shots')
    parser.add_argument('--dataset', type=str, default='caltech256', help='Dataset name')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways')
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project="classical-augmentation-experiments",
        config={
            "seed": args.seed,
            "shots": args.shots,
            "dataset": args.dataset,
            "ways": args.ways,
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": "resnet50",
        }
    )
    
    # Update dataset paths to use arguments
    base_path = f'few_shot_datasets/{args.dataset}/{args.shots}_shot/seed_{args.seed}'
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

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        })

    wandb.finish()
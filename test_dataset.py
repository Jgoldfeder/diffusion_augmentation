import torchvision.transforms as transforms

from CustomDataset import FewShotDataset, ClassicalDataset

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch import nn
import torch


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	train_dataset = FewShotDataset('few_shot_datasets/caltech256/2_shot/seed_41', dataset_type='train')

	print(len(train_dataset))
	print(train_dataset[0])

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	classical_dataset = ClassicalDataset(train_dataset, transform, duplicate_factor=5)

	print(len(classical_dataset))
	print(classical_dataset[0])

	train_loader = DataLoader(classical_dataset, batch_size=32, shuffle=True)
	model = resnet50(weights=ResNet50_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = False
	model.fc = nn.Linear(model.fc.in_features, 256)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(100):
		model.train()
		epoch_loss = 0
		train_correct = 0
		train_total = 0
		
		for images, labels, class_names in train_loader:
			images, labels = images.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()

			_, predicted = torch.max(outputs.data, 1)
			train_total += labels.size(0)
			train_correct += (predicted == labels).sum().item()
		train_accuracy = 100 * train_correct / train_total
		
		avg_loss = epoch_loss / len(train_loader)
		
		print(f'Epoch {epoch+1}/{400}, '
				f'Loss: {avg_loss:.4f}, '
				f'Train Accuracy: {train_accuracy:.2f}%')
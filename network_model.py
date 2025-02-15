from enum import Enum

import torch
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

class ModelType(Enum):
	RESNET50 = 0

def get_resnet50_for_finetune(num_outputs):
	model = resnet50(weights=ResNet50_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = False
	model.fc = Linear(model.fc.in_features, num_outputs)
	return model

def get_model_for_finetune(model_type: ModelType, num_outputs):
	if model_type == ModelType.RESNET50:
		return get_resnet50_for_finetune(num_outputs)
	else:
		raise ValueError(f"Unsupported model: {model_type}")

# TODO run a certain number of epochs of finetuning
def probe_train(model: Module, train_dataset, val_dataset, num_epochs, device):
	model.to(device)
	model.train()
	optimizer = Adam(model.parameters(), lr=0.001)
	criterion = CrossEntropyLoss()

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	val_losses = []
	for epoch in range(num_epochs):
		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			val_loss = 0
			for images, labels in val_loader:
				images = images.to(device)
				labels = labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				val_loss += loss.item()
			val_loss /= len(val_loader)
			val_losses.append(val_loss)
	return val_losses

def evaluate(model, dataset):
	pass

if __name__ == '__main__':
	from dataset_manager import get_dataset_path, FolderDataset

	train_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=True))
	val_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=False))

	my_model = get_model_for_finetune(ModelType.RESNET50, 5)
	val_losses = probe_train(my_model, train_dataset, val_dataset, 20, 'cuda')
	print(val_losses)
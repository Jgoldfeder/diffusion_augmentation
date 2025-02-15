from enum import Enum

import logging
import torch
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

class ModelType(Enum):
	RESNET50 = 'resnet50'

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
	
def get_optimizer(model: Module):
	return Adam(model.parameters(), lr=0.001)

def get_criterion():
	return CrossEntropyLoss()

class ModelResults():
	def __init__(
		self, train_losses: list[float], train_accs: list[float],
		losses: list[float], accs: list[float], preds: list[int], labels: list[int]
	):
		self.train_losses = train_losses
		self.train_accs = train_accs
		self.losses = losses
		self.accs = accs
		self.preds = preds
		self.labels = labels

# TODO run a certain number of epochs of finetuning
def train_and_val(model: Module, train_dataset, val_dataset, num_epochs, device):
	model.to(device)
	optimizer = get_optimizer(model)
	criterion = get_criterion()

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []

	for epoch in range(num_epochs):
		train_losses.append(0)
		train_accs.append(0)
		val_losses.append(0)
		val_accs.append(0)

		model.train()
		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			_, predicted = torch.max(outputs.data, 1)

			train_losses[-1] += loss.item()
			train_accs[-1] += (predicted == labels).sum().item()
		train_losses[-1] /= len(train_loader)
		train_accs[-1] /= len(train_loader)

		model.eval()
		with torch.no_grad():
			for images, labels in val_loader:
				images = images.to(device)
				labels = labels.to(device)

				outputs = model(images)
				loss = criterion(outputs, labels)
				_, predicted = torch.max(outputs.data, 1)

				val_losses[-1] += loss.item()
				val_accs[-1] += (predicted == labels).sum().item()
			val_losses[-1] /= len(val_loader)
			val_accs[-1] /= len(val_loader)

		epoch_info = {
			'epoch': epoch,
			'train_loss': train_losses[-1],
			'train_acc': train_accs[-1],
			'loss': val_losses[-1],
			'acc': val_accs[-1]
		}
		logging.info(f"{epoch_info}")

	# get info for confusion matrix
	val_preds = []
	val_labels = []
	with torch.no_grad():
		for images, labels in val_loader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)

			val_preds.extend(predicted.cpu().numpy())
			val_labels.extend(labels.cpu().numpy())
	return ModelResults(train_losses, train_accs, val_losses, val_accs, val_preds, val_labels)

def train_and_test(model: Module, train_dataset, test_dataset, num_epochs, device):
	return train_and_val(model, train_dataset, test_dataset, num_epochs, device)

def evaluate(model, dataset):
	pass

if __name__ == '__main__':
	from dataset_manager import get_dataset_path, FolderDataset
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	train_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=True))
	val_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=False))

	my_model = get_model_for_finetune(ModelType.RESNET50, 5)
	results = train_and_val(my_model, train_dataset, val_dataset, 20, 'cuda')

	print(results)
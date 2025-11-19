from enum import Enum

import logging
import torch
from torch.nn import CrossEntropyLoss, Linear, Module, Identity
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
import timm
import os 

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ModelType(Enum):
	RESNET50 = 'resnet50'
	VIT224 = 'vit224'
	MOBILENETV2 = 'mobilenetv2'
	VITS = 'vits'  # ViT-Small

def get_vit224_scratch(num_outputs):
	model = timm.create_model("vit_base_patch16_224", pretrained=False)
	if hasattr(model, "head"):
		model.head = Linear(model.head.in_features, num_outputs)
	elif hasattr(model, "classifier"):
		model.classifier = Linear(model.classifier.in_features, num_outputs)
	return model

def get_vit224_for_finetune(num_outputs):
	model = timm.create_model("vit_base_patch16_224", pretrained=True)
	# Freeze all parameters except the head
	for param in model.parameters():
		param.requires_grad = False
	if hasattr(model, "head"):
		model.head = Linear(model.head.in_features, num_outputs)
		model.head.requires_grad = True
	elif hasattr(model, "classifier"):
		model.classifier = Linear(model.classifier.in_features, num_outputs)
		model.classifier.requires_grad = True
	return model

def get_resnet50_scratch(num_outputs):
	model = resnet50(weights=None)
	model.fc = Linear(model.fc.in_features, num_outputs)
	return model

def get_resnet50_for_finetune(num_outputs):
	model = resnet50(weights=ResNet50_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = False
	model.fc = Linear(model.fc.in_features, num_outputs)
	return model

def get_mobilenetv2_scratch(num_outputs):
	model = mobilenet_v2(weights=None)
	model.classifier[1] = Linear(model.classifier[1].in_features, num_outputs)
	return model

def get_mobilenetv2_for_finetune(num_outputs):
	model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
	# Freeze all parameters except the classifier
	for param in model.parameters():
		param.requires_grad = False
	model.classifier[1] = Linear(model.classifier[1].in_features, num_outputs)
	model.classifier[1].requires_grad = True
	return model

def get_vits_scratch(num_outputs):
	model = timm.create_model("vit_small_patch16_224", pretrained=False)
	if hasattr(model, "head"):
		model.head = Linear(model.head.in_features, num_outputs)
	elif hasattr(model, "classifier"):
		model.classifier = Linear(model.classifier.in_features, num_outputs)
	return model

def get_vits_for_finetune(num_outputs):
	model = timm.create_model("vit_small_patch16_224", pretrained=True)
	# Freeze all parameters except the head
	for param in model.parameters():
		param.requires_grad = False
	if hasattr(model, "head"):
		model.head = Linear(model.head.in_features, num_outputs)
		model.head.requires_grad = True
	elif hasattr(model, "classifier"):
		model.classifier = Linear(model.classifier.in_features, num_outputs)
		model.classifier.requires_grad = True
	return model

def get_model_for_finetune(model_type: ModelType, num_outputs):
	if model_type == ModelType.RESNET50:
		return get_resnet50_for_finetune(num_outputs)
	elif model_type == ModelType.VIT224:
		return get_vit224_for_finetune(num_outputs)
	elif model_type == ModelType.MOBILENETV2:
		return get_mobilenetv2_for_finetune(num_outputs)
	elif model_type == ModelType.VITS:
		return get_vits_for_finetune(num_outputs)
	else:
		raise ValueError(f"Unsupported model: {model_type}")

def get_model_for_scratch(model_type: ModelType, num_outputs):
	if model_type == ModelType.RESNET50:
		return get_resnet50_scratch(num_outputs)
	elif model_type == ModelType.VIT224:
		return get_vit224_scratch(num_outputs)
	elif model_type == ModelType.MOBILENETV2:
		return get_mobilenetv2_scratch(num_outputs)
	elif model_type == ModelType.VITS:
		return get_vits_scratch(num_outputs)
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

def train_and_val(model: Module, train_dataset, val_dataset, num_epochs, device, finetune=True):
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
		train_accs[-1] /= len(train_dataset)

		if epoch % 5 == 0:
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
			val_accs[-1] /= len(val_dataset)

		if not finetune:
			checkpoint = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_loss': train_losses[-1],
				'train_acc': train_accs[-1],
				'val_loss': val_losses[-1],
				'val_acc': val_accs[-1]
			}
			os.makedirs('checkpoints', exist_ok=True)
			torch.save(checkpoint, f'checkpoints/epoch_{epoch}.pt')

			if val_accs[-1] > best_val_acc:
				best_val_acc = val_accs[-1]
				torch.save(checkpoint, 'checkpoints/best_model.pt')

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

def train_and_test(model: Module, train_dataset, test_dataset, num_epochs, device, finetune=True):
	return train_and_val(model, train_dataset, test_dataset, num_epochs, device, finetune)

def train(model, train_dataset, num_epochs, device):
	model.to(device)
	optimizer = get_optimizer(model)
	criterion = get_criterion()

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

	train_losses = []
	train_accs = []

	for epoch in range(num_epochs):
		train_losses.append(0)
		train_accs.append(0)

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
		train_accs[-1] /= len(train_dataset)
		
		epoch_info = {
			'epoch': epoch,
			'loss': train_losses[-1],
			'acc': train_accs[-1]
		}
		logging.info(f"{epoch_info}")

	return ModelResults(train_losses, train_accs, [], [], [], [])

def evaluate(model, dataset):
	pass

if __name__ == '__main__':
	from dataset_manager import get_dataset_path, FolderDataset
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

	train_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=True))
	test_dataset = FolderDataset(get_dataset_path('flowers102', 5, 2, 42, train=False))

	# Test all models
	for model_type in [ModelType.RESNET50, ModelType.VIT224, ModelType.MOBILENETV2, ModelType.VITS]:
		logging.info(f"Testing {model_type.value}")
		my_model = get_model_for_finetune(model_type, 5)
		results = train_and_test(my_model, train_dataset, test_dataset, 20, 'cuda:0')
		logging.info(f"Results for {model_type.value}: {results}")
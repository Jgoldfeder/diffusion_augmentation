import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

from AugmentationNode import AugmentationNode

def get_base_transform():
	return Compose([
		ToTensor(),
		Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

def get_classical_transform():
	return Compose([
		Resize(size=(256, 256)),
		RandomCrop(size=(224, 224)),
		ColorJitter(
			brightness=0.4,
			contrast=0.4,
			saturation=0.4,
			hue=0.2 
		),
		RandomHorizontalFlip(p=0.5),
		RandomVerticalFlip(p=0.5), 
		RandomRotation(degrees=10)
	])

class FolderDataset(Dataset):
	def __init__(self, dataset_path):
		self.images: list[Image.Image] = []
		self.labels: list[int] = []
		self.labels_to_class: dict[int, str] = dict()

		for local_class_path in os.listdir(dataset_path):
			class_parts = local_class_path.split('_')
			class_label = int(class_parts[0])
			class_name = '_'.join(class_parts[1:])
			self.labels_to_class[class_label] = class_name

			class_path = os.path.join(dataset_path, local_class_path)

			for img in os.listdir(class_path):
				if img.endswith('.png'):
					img = Image.open(os.path.join(class_path, img))
					self.images.append(img)
					self.labels.append(class_label)

	def get_class_for_label(self, label):
		return self.labels_to_class[label]

	def get_class_for_index(self, index):
		return self.get_class_for_label(self.labels[index])

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

class TreeAugmentedDataset(FolderDataset):
	def __init__(self, dataset_path: str, augmentation_tree: AugmentationNode, num_augmentations_per_image: int):
		super().__init__(dataset_path)

		images_to_add = []
		labels_to_add = []
		for img, label in self:
			for _ in range(num_augmentations_per_image):
				# TODO make this line work
				augmented_img = augmentation_tree.augment(img)
				images_to_add.append(augmented_img)
				labels_to_add.append(label)

		self.images.extend(images_to_add)
		self.labels.extend(labels_to_add)
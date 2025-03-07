import random
import logging
from enum import Enum

import dataset_manager
from dataset_manager import FolderDataset
from image_augmentation_models.augmentation_manager import AugmentationManager, AugmentationType

class ProbabilityLimits(Enum):
	LOW = .3
	HIGH = .7

	def get_random_probability():
		return random.random() * (ProbabilityLimits.HIGH.value - ProbabilityLimits.LOW.value) + ProbabilityLimits.LOW.value

# NOTE think this only works as a balanced tree for now
class BinaryAugmentationNode:
	def __init__(self, augmentation_type: AugmentationType=AugmentationType.NONE):
		self.augmentation_type = augmentation_type
		self.left_probability = 0.0
		self.left: BinaryAugmentationNode = None
		self.right: BinaryAugmentationNode = None

	def get_left_probability(self):
		return self.left_probability
	
	def get_right_probability(self):
		return 1 - self.get_left_probability()

	def generate_augmentation(self, img, class_name):
		logging.info(f'Current node: {self.augmentation_type}')
		if self.augmentation_type == AugmentationType.NONE:
			img = img
		elif self.augmentation_type == AugmentationType.CLASSICAL:
			img = dataset_manager.get_classical_transform()(img)
		else:
			img = AugmentationManager.get_manager(self.augmentation_type).generate_augmentations([img], [class_name])[0]
		
		if not (self.left and self.right):
			return img

		if random.random() < self.get_left_probability():
			return self.left.generate_augmentation(img, class_name)
		else:
			return self.right.generate_augmentation(img, class_name)

	def make_random_tree(self, num_levels):
		self.augmentation_type = AugmentationType.get_random_augmentation()
		self.left_probability = ProbabilityLimits.get_random_probability()
		levels_to_create = num_levels - 1
		if levels_to_create > 0:
			self.left = BinaryAugmentationNode()
			self.right = BinaryAugmentationNode()
			self.left.make_random_tree(levels_to_create)
			self.right.make_random_tree(levels_to_create)

	def str_helper(self, level=0):
		str_rep = ''
		if level == 0:
			str_rep += 'root '
		str_rep += f"({self.augmentation_type}"
		if self.left or self.right:
			str_rep += f" L_prob: {self.get_left_probability():.3f}, R_prob: {self.get_right_probability():.3f}"
		str_rep += ')'
		next_level = level + 1
		spacer = '  ' * next_level
		if self.left:
			str_rep += '\n' + spacer + 'L ' + self.left.str_helper(next_level)
		if self.right:
			str_rep += '\n' + spacer + 'R ' + self.right.str_helper(next_level)
		return str_rep

	def __str__(self):
		return self.str_helper(level=0)

class TreeAugmentedDataset(FolderDataset):
	def __init__(self, dataset_path: str, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
		super().__init__(dataset_path)

		original_size = len(self.images)
		for i in range(original_size): # get them without transforms
			img = self.images[i]
			label = self.labels[i]
			for _ in range(num_augmentations_per_image):
				class_name = self.labels_to_class[label]
				logging.info(f'Generating augmentation for {class_name}')
				augmented_img = augmentation_tree.generate_augmentation(img, class_name)
				self.images.append(augmented_img)
				self.labels.append(label)
				
class TreeAugmentedDatasetFromDataset(FolderDataset):
	def __init__(self, dataset, augmentation_tree: BinaryAugmentationNode, num_augmentations_per_image: int):
		self.images = [img for img in dataset.images]
		self.labels = [label for label in dataset.labels]
		self.labels_to_class = dataset.labels_to_class
		self.base_transform = dataset.base_transform
		self.dataset_path = dataset.dataset_path

		original_size = len(self.images)
		for i in range(original_size): # get them without transforms
			img = self.images[i]
			label = self.labels[i]
			for _ in range(num_augmentations_per_image):
				class_name = self.labels_to_class[label]
				logging.info(f'Generating augmentation for {class_name}')
				augmented_img = augmentation_tree.generate_augmentation(img, class_name)
				self.images.append(augmented_img)
				self.labels.append(label)


if __name__ == '__main__':
	from PIL import Image
	import os

	logging.basicConfig(level=logging.INFO)

	augmentation_tree = BinaryAugmentationNode()
	augmentation_tree.augmentation_type = AugmentationType.SEGMENT

	input_dir = './aug_images/base_images/'
	output_dir = './aug_images/final_images/segment/'

	os.makedirs(output_dir, exist_ok=True)

	base_images = []
	base_classes = []
	# read base images from './aug_images/base_images'
	for filename in os.listdir(input_dir):
		if filename.endswith('.png'):
			base_images.append(Image.open(f'{input_dir}{filename}'))
			base_classes.append(filename.split('.')[0])

	for img in base_images:
		augmented_img = augmentation_tree.generate_augmentation(img, base_classes[base_images.index(img)])
		augmented_img.save(f'{output_dir}{base_classes[base_images.index(img)]}.png')


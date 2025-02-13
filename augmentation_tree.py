import random
from enum import Enum

import dataset_models
from image_augmentation_models.augmentation_manager import AugmentationManager

class AugmentationType(Enum):
	CANNY = 0
	DEPTH = 1
	SEGMENT = 2
	COLOR = 3
	NERF = 4
	CLASSICAL = 5
	NONE = 6

	def get_random_augmentation():
		return random.choice(list(AugmentationType))

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

	def generate_augmentation(self, augmentation_manager: AugmentationManager, img, class_name):
		if self.augmentation_type == AugmentationType.NONE:
			img = img
		elif self.augmentation_type == AugmentationType.CLASSICAL:
			img = dataset_models.get_classical_transform()(img)
		elif self.augmentation_type == AugmentationType.COLOR:
			img = augmentation_manager.color_manager.generate_augmentations([img], [class_name])[0]
		elif self.augmentation_type == AugmentationType.CANNY:
			img = augmentation_manager.canny_manager.generate_augmentations([img], [class_name])[0]
		elif self.augmentation_type == AugmentationType.SEGMENT:
			img = augmentation_manager.segment_manager.generate_augmentations([img], [class_name])[0]
		elif self.augmentation_type == AugmentationType.NERF:
			img = augmentation_manager.nerf_manager.generate_augmentations([img], [class_name])[0]
		elif self.augmentation_type == AugmentationType.DEPTH:
			img = augmentation_manager.depth_manager.generate_augmentations([img], [class_name])[0]
		
		if not (self.left and self.right):
			return img

		if random.random() < self.get_left_probability():
			return self.left.generate_augmentation(augmentation_manager, img, class_name)
		else:
			return self.right.generate_augmentation(augmentation_manager, img, class_name)

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

if __name__ == '__main__':
	from PIL import Image
	import time

	random.seed(time.time())

	node = BinaryAugmentationNode()
	node.make_random_tree(3)
	print(node)

	aug_manager = AugmentationManager()

	# load img from /home/shreyes/diffusion_augmentation/orig_images/0.png
	img = Image.open('/home/shreyes/diffusion_augmentation/orig_images/0.png')

	img = node.generate_augmentation(aug_manager, img, 'tent')

	# save img to /home/shreyes/diffusion_augmentation/aug_images/0.png
	img.save('/home/shreyes/diffusion_augmentation/aug_images/0.png')
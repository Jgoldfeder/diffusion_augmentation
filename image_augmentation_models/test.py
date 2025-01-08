if __name__ == '__main__':
	import os
	import random

	from torchvision.datasets import Caltech256

	from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
	from image_augmentation_models.DepthAugmentation import DepthAugmentationManager
	from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
	from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
	from image_augmentation_models.NerfAugmentation import NerfAugmentationManager

	def save_images(folder, images):
		os.makedirs(folder, exist_ok=True)
		for i, img in enumerate(images):
			img.save(os.path.join(folder, f"{i}.png"))

	# curr_am = CannyAugmentationManager()
	# curr_am = DepthAugmentationManager()
	# curr_am = SegmentAugmentationManager()
	# curr_am = ColorControlNetAugmentationManager()
	curr_am = NerfAugmentationManager()

	dataset = Caltech256(root='./torch', download=True)

	sample_indices = random.sample(range(len(dataset)), 3)
	sample_images = []
	sample_classes = []
	for idx in sample_indices:
		image, label = dataset[idx]
		sample_images.append(image)
		sample_classes.append(f"person standing in outdoors with rainbow")  # Use class_{label} for naming

	# aug_images = curr_am.generate_augmentations(sample_images, sample_classes)
	aug_images = curr_am.generate_augmentations(sample_images)

	save_images("orig_images", sample_images)
	save_images("aug_images", aug_images)
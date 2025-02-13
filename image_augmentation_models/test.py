if __name__ == '__main__':
	import os
	import time
	from PIL import Image

	from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
	from image_augmentation_models.DepthAugmentation import DepthAugmentationManager
	from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
	from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
	from image_augmentation_models.NerfAugmentation import NerfAugmentationManager

	def save_images(folder, images):
		os.makedirs(folder, exist_ok=True)
		for i, img in enumerate(images):
			img.save(os.path.join(folder, f"{i}.png"))

	aug_managers = {
		'canny': CannyAugmentationManager(),
		# 'depth': DepthAugmentationManager(),
		# 'segment': SegmentAugmentationManager(),
		# 'color': ColorControlNetAugmentationManager(),
		# 'nerf': NerfAugmentationManager()
	}

	img_path = './test_images/original.png'
	img = Image.open(img_path)
	sample_images = [img]
	sample_classes = ['tent']

	aug_images = []
	for name, curr_am in aug_managers.items():
		print(f"Testing {name}")
		start_time = time.time()
		aug_images.extend(curr_am.generate_augmentations(sample_images, sample_classes))
		end_time = time.time()
		print(f"Took {end_time - start_time} seconds")

	save_images("orig_images", sample_images)
	save_images("aug_images", aug_images)
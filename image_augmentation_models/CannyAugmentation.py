import time

import cv2
import numpy as np
from PIL import Image
from controlnet_aux import CannyDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class CannyAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda:0"):
		super().__init__("lllyasviel/sd-controlnet-canny", control_net_device)
		self.canny = CannyDetector()

	def preprocess_image(self, original_image):
		img = super().preprocess_image(original_image)

		np_img = np.array(img)
		low_threshold = 100
		high_threshold = 200

		edges = cv2.Canny(np_img, low_threshold, high_threshold)
		edges = edges[:, :, None]
		edges = np.concatenate([edges, edges, edges], axis=2)

		processed_image = Image.fromarray(edges)
		return processed_image

if __name__ == '__main__':
	aug_manager = CannyAugmentationManager()
	print('Created Canny Aug Manager')

	img_path = './test_images/original.png'
	img = Image.open(img_path)
	print('loaded image')

	input_images = [img] * 5
	input_prompts = ['tent'] * 5

	start_time = time.time()
	images = aug_manager.generate_augmentations(input_images, input_prompts)
	end_time = time.time()
	print('Took {} seconds'.format(end_time - start_time))

	for i, image in enumerate(images):
		image.save('./test_images/canny_' + str(i) + '.png')


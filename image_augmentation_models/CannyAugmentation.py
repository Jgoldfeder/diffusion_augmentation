import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class CannyAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda"):
		super().__init__(control_net_device)
		self.canny = CannyDetector()
		self.controlnet_model = "lllyasviel/sd-controlnet-canny"

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

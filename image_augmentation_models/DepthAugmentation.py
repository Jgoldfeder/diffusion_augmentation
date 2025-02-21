import os
import numpy as np
import torch
from PIL import Image
from controlnet_aux import MidasDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class DepthAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda:0"):
		super().__init__("lllyasviel/sd-controlnet-depth", control_net_device)
		self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

	def preprocess_image(self, original_image):
		img = super().preprocess_image(original_image)
		return self.midas(img)

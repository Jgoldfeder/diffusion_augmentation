import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MidasDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class DepthAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda"):
		super().__init__(control_net_device)
		self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
		self.controlnet_model = "lllyasviel/sd-controlnet-depth"

	def preprocess_image(self, original_image):
		img = super().preprocess_image(original_image)
		return self.midas(img)

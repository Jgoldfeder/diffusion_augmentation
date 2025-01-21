import os
import numpy as np
import torch
from PIL import Image
from controlnet_aux import SamDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class SegmentAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda"):
		super().__init__(control_net_device)
		self.sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
		self.controlnet_model = "lllyasviel/sd-controlnet-seg"

	def preprocess_image(self, original_image):
		img = super().preprocess_image(original_image)
		return self.sam(img)
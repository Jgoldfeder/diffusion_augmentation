import os
import numpy as np
import torch
from PIL import Image
from controlnet_aux import SamDetector

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager

class SegmentAugmentationManager(ControlNetAugmentationManager):
	def __init__(self, control_net_device="cuda:0"):
		super().__init__("lllyasviel/sd-controlnet-seg", control_net_device)
		self.sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

	def preprocess_image(self, original_image):
		img = super().preprocess_image(original_image)
		return self.sam(img)
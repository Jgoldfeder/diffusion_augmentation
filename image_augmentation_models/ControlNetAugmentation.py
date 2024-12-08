import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

class ControlNetAugmentationManager:
	def __init__(self, control_net_device="cuda"):
		self.control_net_device = control_net_device

	def preprocess_image(self, img):
		return img.convert("RGB").resize((512, 512))
		# NOTE more preprocessing needs to occur specific to if we
		# use canny, midas, seg, etc.
		# but this is the base we always do

	def generate_augmentations(self, images, classes):
		augmented = []

		for img, img_class in zip(images, classes):
			processed_image = self.preprocess_image(img)

			controlnet = ControlNetModel.from_pretrained(
				self.controlnet_model,
				torch_dtype=torch.float16
			).to(self.control_net_device)

			pipe = StableDiffusionControlNetPipeline.from_pretrained(
				"runwayml/stable-diffusion-v1-5",
				controlnet=controlnet,
				torch_dtype=torch.float16
			).to(self.control_net_device)

			pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
			pipe.enable_xformers_memory_efficient_attention()
			
			class_prompt = img_class

			prompt = [f"Extremely Realistic, Photorealistic, Clear Image, Real World, {class_prompt}"]
			negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]
			generator = [torch.Generator(device=self.control_net_device).manual_seed(2) for _ in range(len(prompt))]

			output = pipe(
				prompt,
				processed_image,
				negative_prompt=negative_prompt * len(prompt),
				generator=generator,
				num_inference_steps=20,
			)

			augmented.append(output.images[0])

		return augmented

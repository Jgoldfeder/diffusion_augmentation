import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import random

class ControlNetAugmentationManager:
	def __init__(self, controlnet_model, control_net_device="cuda:0"):
		self.control_net_device = control_net_device
		self.controlnet_model = controlnet_model

		# NOTE once we find a way to optimize gpu space we can move 
		# the following back into the init function which will save a lot of
		# time generating images
		# Load ControlNet and pipeline
		# controlnet = ControlNetModel.from_pretrained(
		# 	controlnet_model,
		# 	torch_dtype=torch.float16
		# ).to(self.control_net_device)

		# self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
		# 	"runwayml/stable-diffusion-v1-5",
		# 	controlnet=controlnet,
		# 	torch_dtype=torch.float16,
		# 	safety_checker=None
		# ).to(self.control_net_device)

		# self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
		# self.pipe.enable_xformers_memory_efficient_attention()

	def preprocess_image(self, img):
		return img.convert("RGB").resize((512, 512))
		# NOTE more preprocessing needs to occur specific to if we
		# use canny, midas, seg, etc.
		# but this is the base we always do

	def generate_augmentations(self, images, classes):
		augmented = []

		controlnet = ControlNetModel.from_pretrained(
			self.controlnet_model,
			torch_dtype=torch.float16
		).to(self.control_net_device)

		pipe = StableDiffusionControlNetPipeline.from_pretrained(
			"models/runwayml-stable-diffusion-v1-5",
			controlnet=controlnet,
			torch_dtype=torch.float16,
			safety_checker=None
		).to(self.control_net_device)

		pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.enable_xformers_memory_efficient_attention()

		for img, img_class in zip(images, classes):
			processed_image = self.preprocess_image(img)
			
			class_prompt = img_class

			prompt = [f"Extremely Realistic, Photorealistic, Clear Image, Real World, {class_prompt}"]
			negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]
			generator = torch.Generator(device=self.control_net_device).manual_seed(random.randint(0, 1000000))

			output = pipe(
				prompt,
				processed_image,
				negative_prompt=negative_prompt * len(prompt),
				generator=generator,
				num_inference_steps=20,
			)

			augmented.append(output.images[0])

		return augmented

	# NOTE this function should not yet be used since we don't have enough GPU space
	def generate_augmentations_parallel(self, images, classes):
		raise NotImplementedError
		processed_images = [self.preprocess_image(img) for img in images]

		prompts = [f"Extremely Realistic, Photorealistic, Clear Image, Real World, {cls}" for cls in classes]
		negative_prompts = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompts)

		generator = torch.Generator(device=self.control_net_device)

		output = self.pipe(
			prompts,
			processed_images,
			negative_prompt=negative_prompts,
			generator=generator,
			num_inference_steps=20,
		)

		return output.images

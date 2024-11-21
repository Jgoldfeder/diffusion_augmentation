import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MidasDetector

class DepthAugmentationManager:
    def __init__(self, control_net_device="cuda"):
        self.control_net_device = control_net_device
        self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        self.controlnet_model = "lllyasviel/sd-controlnet-depth"

    def preprocess_image(self, image_path):
        """
        Processes an image using Midas depth detection.
        """
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        return self.midas(img)

    def generate_augmentations(self, image_paths):
        """
        Generate depth-based augmentations for the specified images.
        Returns dictionary containing augmented images.
        """
        depth_augmented = {}

        for image_path in image_paths:
            processed_image = self.preprocess_image(image_path)

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
            
            print(f"<LOG> Image path: {image_path}")
            class_prompt = image_path.split('/')[-2]
            if len(class_prompt.split('.')) > 1:
                class_prompt = class_prompt.split('.')[1].replace('-', ' ')
            print(f"<LOG> Class prompt: {class_prompt}")

            prompt = [f"{class_prompt}"]
            negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]
            generator = [torch.Generator(device=self.control_net_device).manual_seed(2)]

            output = pipe(
                prompt,
                processed_image,
                negative_prompt=negative_prompt * len(prompt),
                generator=generator,
                num_inference_steps=20,
            )

            depth_augmented[image_path] = output.images[0]

        return depth_augmented

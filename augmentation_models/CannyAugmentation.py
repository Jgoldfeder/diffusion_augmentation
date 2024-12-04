import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

class CannyAugmentationManager:
    def __init__(self, control_net_device="cuda"):
        self.control_net_device = control_net_device
        self.canny = CannyDetector()
        self.controlnet_model = "lllyasviel/sd-controlnet-canny"

    def preprocess_image(self, image_path):
        """
        Processes an image with the specified preprocessor type.
        """
        img = Image.open(image_path).convert("RGB").resize((512, 512))

        np_img = np.array(img)
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(np_img, low_threshold, high_threshold)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        processed_image = Image.fromarray(edges)
        
        return processed_image

    def generate_augmentations(self, image_paths):
        """
        Generate augmentations using only Canny edge detection.
        Returns dictionary containing augmented images.
        """
        canny_augmented = {}

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

            canny_augmented[image_path] = output.images[0]

        return canny_augmented

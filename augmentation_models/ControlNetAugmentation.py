import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector, MidasDetector, SamDetector

class ControlNetAugmentationManager:
    def __init__(self, control_net_device="cuda"):
        self.control_net_device = control_net_device
        self.canny = CannyDetector()
        self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        self.sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
        
        self.controlnet_models = {
            "Canny": "lllyasviel/sd-controlnet-canny",
            "Midas": "lllyasviel/sd-controlnet-depth",
            "Segmentation": "lllyasviel/sd-controlnet-seg"
        }

    def preprocess_image(self, image_path, preprocessor_type):
        """
        Processes an image with the specified preprocessor type.
        """
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        
        if preprocessor_type == "Canny":
            np_img = np.array(img)
            low_threshold = 100
            high_threshold = 200
            edges = cv2.Canny(np_img, low_threshold, high_threshold)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            processed_image = Image.fromarray(edges)
        elif preprocessor_type == "Midas":
            processed_image = self.midas(img)
        elif preprocessor_type == "Segmentation":
            processed_image = self.sam(img)
        else:
            raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")
        
        return processed_image

    def generate_augmentations(self, image_paths):
        """
        Generate augmentations for the specified classes.
        Returns dictionaries containing augmented images for each conditioning type.
        """
        canny_augmented = {}
        depth_augmented = {}
        segmentation_augmented = {}

        for image_path in image_paths:
            preprocessor_types = ["Canny", "Midas", "Segmentation"]
            processed_images = {}

            for preprocessor_type in preprocessor_types:
                processed_image = self.preprocess_image(image_path, preprocessor_type)
                processed_images[preprocessor_type] = processed_image

            for preprocessor_type, model_id in self.controlnet_models.items():
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                ).to(self.control_net_device)

                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16
                ).to(self.control_net_device)

                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_xformers_memory_efficient_attention()

                class_prompt = image_path.split('/')[-1].split('.')[1].replace('-', ' ')
                prompt = [f"{class_prompt}, best quality, extremely detailed"]
                negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]
                generator = [torch.Generator(device=self.control_net_device).manual_seed(2) for _ in range(len(prompt))]

                output = pipe(
                    prompt,
                    processed_images[preprocessor_type],
                    negative_prompt=negative_prompt * len(prompt),
                    generator=generator,
                    num_inference_steps=20,
                )

                # Store generated images in appropriate dictionary
                if preprocessor_type == "Canny":
                    canny_augmented[image_path] = output.images[0]
                elif preprocessor_type == "Midas":
                    depth_augmented[image_path] = output.images[0]
                elif preprocessor_type == "Segmentation":
                    segmentation_augmented[image_path] = output.images[0]

        return canny_augmented, depth_augmented, segmentation_augmented

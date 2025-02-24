import sys
import os
from PIL import Image
import torch
from typing import Dict

# Add color controlnet to path
color_controlnet_dir = os.path.join(os.getcwd(), 'color_controlnet')
sys.path.append(color_controlnet_dir)

# Color ControlNet imports
from color_controlnet.diffusers import (
    ControlNetModel,
    LineartDetector,
    StableDiffusionImg2ImgControlNetPalettePipeline,
    UniPCMultistepScheduler
)
from color_controlnet.infer_palette import (
    get_cond_color,
    show_anns,
    image_grid,
    HWC3,
    resize_in_buckets,
    SAMImageAnnotator
)
from color_controlnet.infer_palette_img2img import control_color_augment

class ColorControlNetAugmentationManager:
    def __init__(self, color_control_device: str = "cuda:0"):
        self.color_control_device = color_control_device
        self.color_control_model = self._initialize_color_controlnet_models()

    def _initialize_color_controlnet_models(self):
        """Initialize all models and detectors used in the pipeline"""
        print('Loading Color Control model...')
        color_control = {}

        controlnet = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()
        adapter = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()

        sam_annotator = SAMImageAnnotator()

        model_ckpt = "./models/color_img2img_palette.pt"
        model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

        # assign the weights of the controlnet and adapter separately
        controlnet_sd = {}
        adapter_sd = {}
        for k in model_sd.keys():
            if k.startswith("controlnet"):
                controlnet_sd[k.replace("controlnet.", "")] = model_sd[k]
            if k.startswith("adapter"):
                adapter_sd[k.replace("adapter.", "")] = model_sd[k]


        controlnet.load_state_dict(controlnet_sd, strict=True)
        if adapter is not None:
            adapter.load_state_dict(adapter_sd, strict=False)


        pipe = StableDiffusionImg2ImgControlNetPalettePipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.color_control_device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        color_control['pipe'] = pipe
        color_control['sam_annotator'] = sam_annotator
        color_control['adapter'] = adapter 

        return color_control

    def generate_augmentations(self, images, classes):
        augmented_images = []
        
        for original_image, class_name in zip(images, classes):
            img = original_image.convert("RGB").resize((512, 512))
            
            print(f"<LOG> Class name: {class_name}")

            color_augmented = control_color_augment(
                img, 
                self.color_control_model['adapter'],
                self.color_control_model['pipe'],
                class_name,
                self.color_control_model['sam_annotator'],
                1,
                self.color_control_device
            )
            
            augmented_images.append(color_augmented[0])
            
        return augmented_images
import sys
import os

import torch

color_controlnet_dir = os.path.join("/home/shreyes", 'diffusion_augmentation', 'color_controlnet')
sys.path.append(color_controlnet_dir)

from color_controlnet.diffusers import ControlNetModel, LineartDetector, StableDiffusionImg2ImgControlNetPalettePipeline
from color_controlnet.diffusers import UniPCMultistepScheduler
from color_controlnet.infer_palette import get_cond_color, show_anns, image_grid, HWC3, resize_in_buckets, SAMImageAnnotator
from color_controlnet.infer_palette_img2img import control_color_augment

def initialize_color_controlnet_models(color_control_device):
    """
    Initialize all models and detectors used in the pipeline
    Returns:
        control_net: dict, containing all ControlNet models and detectors
        llava: dict, containing the LLAVA model and processor
        zero123: dict, containing the Zero123 model and Carvekit interf`ace
        color_control: dict, containing the Color Control model and SAM annotator

    """

    # Color Control model
    print('Loading Color Control model...')
    color_control = {}

    controlnet = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()
    adapter = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()

    sketch_method = "skmodel"
    sam_annotator = SAMImageAnnotator()

    model_ckpt = f"./models/color_img2img_palette.pt"
    model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

    # assign the weights of the controlnet and adapter separately
    controlnet_sd = {}
    adapter_sd = {}
    for k in model_sd.keys():
        if k.startswith("controlnet"):
            controlnet_sd[k.replace("controlnet.", "")] = model_sd[k]
        if k.startswith("adapter"):
            adapter_sd[k.replace("adapter.", "")] = model_sd[k]

    msg_control = controlnet.load_state_dict(controlnet_sd, strict=True)
    print(f"msg_control: {msg_control} ")
    if adapter is not None:
        msg_adapter = adapter.load_state_dict(adapter_sd, strict=False)
        print(f"msg_adapter: {msg_adapter} ")

    # define the inference pipline
    # sdv15_path = "/home/pat/diffusion_augmentation/color_controlnet/model_configs/sd15_config.json"
    pipe = StableDiffusionImg2ImgControlNetPalettePipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(color_control_device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    color_control['pipe'] = pipe
    color_control['sam_annotator'] = sam_annotator
    color_control['adapter'] = adapter 

    return color_control
import os
import sys
#if color doesn't work ensure that you uninstall diffusers and use the color controlnet version instead

zero123_dir = os.path.join("/home/pat", 'diffusion_augmentation', 'zero123')
sys.path.append(zero123_dir)

# controlnet_dir = os.path.join("/home/pat", 'diffusion_augmentation', 'controlnet')
# sys.path.append(controlnet_dir)

controlnet_dir = os.path.join("/home/pat", 'diffusion_augmentation', 'color_controlnet')
sys.path.append(controlnet_dir)

import torch
import numpy as np
import cv2
from PIL import Image
import random
import einops
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, LlavaForConditionalGeneration, SamModel, AutoImageProcessor, DPTForDepthEstimation
from transformers import pipeline
from controlnet_old.annotator.util import resize_image, HWC3
from controlnet_old.annotator.canny import CannyDetector
from controlnet_old.annotator.uniformer import UniformerDetector
from controlnet_old.annotator.midas import MidasDetector
from omegaconf import OmegaConf
from controlnet_old.cldm.model import create_model, load_state_dict
from controlnet_old.cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything

# Zero123 imports
# from zero123.nerf import load_model_from_config, generate_angles
from zero123.ldm.util import create_carvekit_interface

# Color Control imports
from color_controlnet.diffusers import ControlNetModel, LineartDetector, StableDiffusionImg2ImgControlNetPalettePipeline
from color_controlnet.diffusers import UniPCMultistepScheduler
from color_controlnet.infer_palette import get_cond_color, show_anns, image_grid, HWC3, resize_in_buckets, SAMImageAnnotator
from color_controlnet.infer_palette_img2img import control_color_augment


control_net_device = 'cuda:1'
zero123_device = 'cuda:1'
color_control_device = 'cuda:1'


def initialize_models():
    """
    Initialize all models and detectors used in the pipeline
    Returns:
        control_net: dict, containing all ControlNet models and detectors
        llava: dict, containing the LLAVA model and processor
        zero123: dict, containing the Zero123 model and Carvekit interface
        color_control: dict, containing the Color Control model and SAM annotator

    """

    # Initialize ControlNet models
    print('Loading ControlNet models...')
    control_net = {}
    model_names = ['control_v11p_sd15_canny', 'control_v11f1p_sd15_depth', 'control_v11p_sd15_seg']
    models = {}
    for name in model_names:
        model = create_model(f'./models/{name}.yaml').cpu()
        model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location=control_net_device), strict=False)
        model.load_state_dict(load_state_dict(f'./models/{name}.pth', location=control_net_device), strict=False)
        models[name] = model.to(control_net_device)

    # Initialize Control Netdetectors
    apply_canny = CannyDetector()
    apply_depth = MidasDetector()
    apply_seg = UniformerDetector()
    detectors = {'Canny': apply_canny, 'Depth': apply_depth, 'Segmentation': apply_seg}
    
    control_net['models'] = models
    control_net['detectors'] = detectors

    # Initialize LLAVA model
    print('Loading LLAVA model...')
    llava = {}
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_id)
    llava['processor'] = processor
    llava['model'] = llava_model

    # Zero123 models
    print('Loading Zero123 models...')
    zero123 = {}
    config_path = './models/sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config_path)

    model_path = "./models/105000.ckpt"
    model = load_model_from_config(config, model_path, zero123_device)
    model = model.to(zero123_device)

    # print('Creating Carvekit interface...')
    carvekit_interface = create_carvekit_interface()

    zero123['model'] = model
    zero123['carvekit_interface'] = carvekit_interface 


    # Color Control model
    print('Loading Color Control model...')
    color_control = {}

    controlnet = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()
    adapter = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()

    sketch_method = "skmodel"
    sam_annotator = SAMImageAnnotator()

    model_ckpt = f"./model_configs/color_img2img_palette.pt"
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

    return control_net, llava, zero123, color_control

def control_augment(control_net, device, det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        input_image = np.array(input_image)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape

        if det == 'Canny':
            detected_map = control_net['detectors']['Canny'](input_image, low_threshold, high_threshold)
            model = control_net['models']['control_v11p_sd15_canny']
        elif det == 'Depth':
            detected_map, _ = control_net['detectors']['Depth'](input_image)
            model = control_net['models']['control_v11f1p_sd15_depth']
        elif det == 'Segmentation':
            detected_map = control_net['detectors']['Segmentation'](input_image)
            model = control_net['models']['control_v11p_sd15_seg']
        else:
            raise ValueError(f"Unknown detection type: {det}")

        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        ddim_sampler = DDIMSampler(model)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return detected_map, results[0]

import colorsys
def random_color_augmentation(image, num_colors=6):
        # Convert the image to grayscale
    grayscale = image.convert('L')
    
    # Convert grayscale image to numpy array
    gray_array = np.array(grayscale)
    
    # Generate random colors
    colors = [colorsys.hsv_to_rgb(np.random.random(), 
                                  np.random.uniform(0.5, 1.0), 
                                  np.random.uniform(0.5, 1.0)) for _ in range(num_colors)]
    
    # Create an empty array for the colored image
    colored_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.float32)
    
    # Normalize the gray values to be between 0 and 1
    normalized_gray = gray_array / 255.0
    
    # Apply the color gradient
    for i in range(num_colors - 1):
        mask = ((normalized_gray >= i / (num_colors - 1)) & 
                (normalized_gray < (i + 1) / (num_colors - 1)))
        
        t = (normalized_gray[mask] - i / (num_colors - 1)) * (num_colors - 1)
        
        colored_array[mask] = (1 - t[:, np.newaxis]) * colors[i] + t[:, np.newaxis] * colors[i+1]
    
    # Handle the last interval
    mask = (normalized_gray >= (num_colors - 1) / (num_colors - 1))
    colored_array[mask] = colors[-1]
    
    # Convert to uint8
    colored_array = (colored_array * 255).astype(np.uint8)
    
    # Convert the numpy array back to a PIL Image
    colored_image = Image.fromarray(colored_array)
    
    return colored_image


def auto_augment(image, class_label, models):
    pil_image = to_pil_image(image) if not isinstance(image, Image.Image) else image
    
    control_net, llava, zero123, color_control = models
    # If you want to use LLAVA for captioning, uncomment these lines:
    # prompt = "USER: <image>\nDescribe the image in detail in as many words as possible. Describe the colors, what's in focus, out of focus, on the ground, and in the sky. \nASSISTANT:"
    # inputs = llava['processor'](prompt, images=[pil_image], return_tensors="pt")
    # generated_ids = llava['model'].generate(**inputs, max_length=100)
    # caption = llava['processor'].batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    caption = "a tent in the middle of a field"
    print(f"Generated caption: {caption}")
    
    augmented_images = {}
    preprocessed_images = {}

    # Apply Control Net augmentation
    for det_type in ['Canny', 'Depth', 'Segmentation']:
        print(f"Processing {det_type}...")
        preprocessed, augmented = control_augment(
            control_net=control_net,
            device=control_net_device,
            det=det_type,
            input_image=pil_image,
            prompt=caption,
            a_prompt="clear image, photorealistic",
            n_prompt="multiple, mushed, low quality, cropped, worst quality",
            num_samples=1,
            image_resolution=512,
            detect_resolution=512,
            ddim_steps=20,
            guess_mode=False,
            strength=1.0,
            scale=7.5,
            seed=-1,
            eta=0.0,
            low_threshold=100,
            high_threshold=200,
            
        )
        augmented_images[det_type] = Image.fromarray(augmented)
        preprocessed_images[det_type] = Image.fromarray(preprocessed)

    #Apply random color augmentation
    augmented_images['RandomColor'] = random_color_augmentation(augmented_images[det_type])

    # Apply Zero123 augmentations
    angles = [
        ("front", 0, 0, 0),
        ("left", 0, -15, 0),
        ("right", 0, 15, 0),
        ("above", -15, 0, 0),
        ("below", 15, 0, 0),
        ("behind", 0, 180, 0),
    ]

    # Returns the preprocessed image and then a list of augmented images of type Image
    preprocessed_zero, augmented_zero = generate_angles(image, angles, zero123['model'], zero123['carvekit_interface'], zero123_device, ddim_steps=50, scale=3.0, n_samples=1)
    
    preprocessed_images['Zero123'] = preprocessed_zero

    #right now will only return the first image 
    augmented_images['Zero123'] = augmented_zero[0]

    # Apply Color Control augmentation returns a list of images
    color_augmented = control_color_augment(pil_image, color_control['adapter'], color_control['pipe'], caption, color_control['sam_annotator'], 1, color_control_device)

    augmented_images['ColorControl'] = color_augmented[0]
   
    return pil_image, preprocessed_images, augmented_images, class_label, caption

if __name__ == "__main__":
    models = initialize_models()
    example_image = Image.open("./test_images/original.png")
    example_class = "tent"
    
    original, preprocessed_dict, augmented_dict, label, caption = auto_augment(example_image, example_class, models)
    

    for det_type in ['Canny', 'Depth', 'Segmentation', 'RandomColor', 'Zero123', 'ColorControl']:
        augmented_dict[det_type].save(f"./test_images/augmented_{det_type.lower()}.png")
        try:
            preprocessed_dict[det_type].save(f"./test_images/preprocessed_{det_type.lower()}.png")
        except KeyError:
            pass
    print(f"Class: {label}")
    print(f"Caption: {caption}")
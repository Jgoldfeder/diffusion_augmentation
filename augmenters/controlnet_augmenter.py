
import sys
import os

controlnet_dir = os.path.join("/home/shreyes", 'diffusion_augmentation', 'controlnet')
sys.path.append(controlnet_dir)

import torch
import numpy as np
import cv2
import random
import einops
from controlnet.annotator.util import resize_image, HWC3
from controlnet.annotator.canny import CannyDetector
from controlnet.annotator.uniformer import UniformerDetector
from controlnet.annotator.midas import MidasDetector

from controlnet.cldm.model import create_model, load_state_dict
from controlnet.cldm.ddim_hacked import DDIMSampler as ControlNetDDIMSampler
from pytorch_lightning import seed_everything

def initialize_models(control_net_device):
    # Initialize ControlNet models
    print('Loading ControlNet models...')
    control_net = {}
    model_names = ['control_v11p_sd15_canny', 'control_v11f1p_sd15_depth', 'control_v11p_sd15_seg']
    models = {}
    for name in model_names:
        model = create_model(f'./model_configs/{name}.yaml').cpu()
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
    return control_net


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
            # print(control_net['detectors']['Depth'](input_image))
            detected_map = control_net['detectors']['Depth'](input_image)
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

        ddim_sampler = ControlNetDDIMSampler(model)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return detected_map, results[0]
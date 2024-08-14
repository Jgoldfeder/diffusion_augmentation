import torch
import numpy as np
import cv2
from PIL import Image
import random
import einops
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, LlavaForConditionalGeneration, SamModel, AutoImageProcessor, DPTForDepthEstimation
from transformers import pipeline
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from annotator.midas import MidasDetector

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything


# Initialize ControlNet models
model_names = ['control_v11p_sd15_canny', 'control_v11f1p_sd15_depth', 'control_v11p_sd15_seg']
models = {}
for name in model_names:
    model = create_model(f'./models/{name}.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{name}.pth', location='cuda'), strict=False)
    models[name] = model.cuda()




# Initialize detectors

apply_canny = CannyDetector()
apply_depth = MidasDetector()
apply_seg = UniformerDetector()

# Initialize LLAVA model
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(model_id)


def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        # input_image is pil image need to change back to numpy
        input_image = np.array(input_image)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        if det == 'Canny':
           
            detected_map = apply_canny(input_image, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            # Choose model and sampler
            model = models['control_v11p_sd15_canny']
            ddim_sampler = DDIMSampler(models['control_v11p_sd15_canny'])  # Using Canny model for sampling

        elif det == 'Depth':
            detected_map, _ = apply_depth(input_image)
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            depth = Image.fromarray(detected_map)
            depth.save(f"./test/preprocessed_depth.png")
            # print(models.keys())

            # Choose model and sampler
            model = models['control_v11f1p_sd15_depth']
            ddim_sampler = DDIMSampler(models['control_v11f1p_sd15_depth'])  # Using Depth Anything model for sampling


        elif det == 'Segmentation':
            detected_map = apply_seg(input_image)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
            # print(type(detected_map))
            # show_masks_on_image(input_image, masks)
            Image.fromarray(detected_map).save(f"./test/preprocessed_segment.png")

            # Choose model and sampler
            model = models['control_v11p_sd15_seg']
            ddim_sampler = DDIMSampler(models['control_v11p_sd15_seg'])
        else:
            raise ValueError(f"Unknown detection type: {det}")

        img = resize_image(HWC3(np.array(input_image)), image_resolution)
        H, W, C = img.shape

        

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
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

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        #save the result
        Image.fromarray(results[0]).save(f"./test/augmented_{det}.png")
    return detected_map, results[0]  # Return both the preprocessed map and the augmented image

def augment(image, class_label, det_type):

    # Step 1: Generate caption using LLAVA
    pil_image = to_pil_image(image) if not isinstance(image, Image.Image) else image
    # prompt = "USER: <image>\nDescribe the image in detail in as many words as possible. Describe the colors, what's in focus, out of focus, on the ground, and in the sky. \nASSISTANT:"
    # inputs = processor(prompt, images=[pil_image], return_tensors="pt")
    
    # generated_ids = llava_model.generate(**inputs, max_length=100)
    # caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    caption = "a tent in the middle of a field"
    print(f"Generated caption: {caption}")
    
    
    # Step 3: Apply ControlNet with different detection types
    augmented_images = {}
    preprocessed_images = {}
    for det_type in ['Canny', 'Depth', 'Segmentation']:
        print(f"Processing {det_type}...")
        preprocessed, augmented = process(
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
            high_threshold=200
        )
        augmented_images[det_type] = Image.fromarray(augmented)
        preprocessed_images[det_type] = Image.fromarray(preprocessed)

    
    return image, preprocessed_images, augmented_images, class_label, caption


# Example usage
if __name__ == "__main__":
    # Load an example image (replace with your own image loading logic)
    example_image = Image.open("/home/pat/diffusion_augmentation/torch/sun397_og/SUN397/t/tent/outdoor/sun_atybcwhfakdbxhlg.jpg")
    example_image.save("./test/original.png")
    example_class = "tent"
    
    original, preprocessed_dict, augmented_dict, label, caption = augment(example_image, example_class)
    
    # Save or display results
    original.save("./test/original.png")
    for det_type in ['Canny', 'Depth', 'Segmentation']:
        preprocessed_dict[det_type].save(f"./test/preprocessed_{det_type.lower()}.png")
        augmented_dict[det_type].save(f"./test/augmented_{det_type.lower()}.png")
    print(f"Class: {label}")
    print(f"Caption: {caption}")




     
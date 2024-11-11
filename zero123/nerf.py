import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from contextlib import nullcontext
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, instantiate_from_config
from omegaconf import OmegaConf
import math
import cv2
import PIL

def load_model_from_config(config, ckpt, device, verbose=False):
    """
    Loads a model from a given configuration and checkpoint.
    
    :param config: The configuration file.
    :param ckpt: The checkpoint path.
    :param device: The device to load the model onto.
    :param verbose: If True, prints additional info about missing/unexpected keys.
    :return: The loaded model.
    """
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')

    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y:y+h, x:x+w, :]
    image = PIL.Image.fromarray(np.array(image))
    
    # resize image such that long edge is 512
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)
    
    return image

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result
def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    # print('old input_im:', input_im.size)
    # start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    # print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    # print('new input_im:', lo(input_im))

    return input_im
def preprocess_image(carvekit, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    # start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(carvekit, input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    # print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    # print('new input_im:', lo(input_im))

    return input_im
import torch
import math
from torchvision import transforms
from PIL import Image, ImageOps

@torch.no_grad()
def generate_angles(input_image, angles, model, carvekit_interface, device, 
                   precision='autocast', h=256, w=256, ddim_steps=50, scale=3.0, 
                   n_samples=1, ddim_eta=1.0):
    """
    Generates augmented images from specified angles.

    :param input_image: PIL Image.
    :param angles: List of tuples (name, x_angle, y_angle, z_angle).
    :param model: Loaded LatentDiffusion model.
    :param carvekit_interface: Carvekit interface for preprocessing.
    :param device: Torch device (e.g., 'cuda').
    :param precision: Precision mode ('autocast' or other).
    :param h: Target height for resizing (must be divisible by 8).
    :param w: Target width for resizing (must be divisible by 8).
    :param ddim_steps: Number of DDIM sampling steps.
    :param scale: Guidance scale for classifier-free guidance.
    :param n_samples: Number of samples to generate per angle.
    :param ddim_eta: DDIM eta parameter.
    :return: Tuple (preprocessed_image_pil, list of augmented PIL Images).
    """
    sampler = DDIMSampler(model, device, precision=precision)
    # 1. Preprocess the input image using Carvekit
    preprocessed_image = preprocess_image(carvekit_interface, input_image, preprocess=True)
    
    #get pil image for preprocessed image
    preprocessed_image_pil = Image.fromarray((preprocessed_image * 255).astype(np.uint8))
    # 4. Convert PIL Image to tensor and normalize to [-1, 1]
    input_tensor = transforms.ToTensor()(preprocessed_image).unsqueeze(0).to(device)
    input_tensor = input_tensor * 2 - 1  # Normalize to [-1, 1]
    
    # 5. Encode the input image to get its latent representation
    input_image_latent = model.get_first_stage_encoding(model.encode_first_stage(input_tensor))
    
    # 6. Initialize list to hold augmented images
    augmented_images = []
    
    # 7. Iterate over each specified angle and generate augmented images
    for name, x_angle, y_angle, z_angle in angles:
        print(f'Generating {name} view with angles x={x_angle}, y={y_angle}, z={z_angle}')
        
        # 7.1. Prepare Conditioning
        # Get learned conditioning and incorporate angles
        c = model.get_learned_conditioning(input_tensor).tile(n_samples, 1, 1)
        
        # Create tensor T from angles
        T = torch.tensor([math.radians(x_angle),
                          math.sin(math.radians(y_angle)),
                          math.cos(math.radians(y_angle)),
                          z_angle], dtype=torch.float32).to(device)
        T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        
        # Concatenate conditioning and angles
        c = torch.cat([c, T], dim=-1)  # Shape: [n_samples, original_cond_dim + 4]
        
        # Project conditioning
        c = model.cc_projection(c)
        
        # 7.2. Prepare Conditioning Dictionary
        cond = {
            'c_crossattn': [c],
            'c_concat': [input_image_latent.repeat(n_samples, 1, 1, 1)]
        }
        
        # 7.3. Prepare Unconditional Conditioning (for classifier-free guidance)
        if scale != 1.0:
            uc = {
                'c_crossattn': [torch.zeros_like(c).to(device)],
                'c_concat': [torch.zeros(n_samples, 4, h // 8, w // 8).to(device)]
            }
        else:
            uc = None
        
        # 7.4. Define Sampling Shape
        shape = [4, h // 8, w // 8]  # [channels, height, width]
        
        # 7.5. Perform Sampling
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            x_T=None
        )
        print(f"Sampled tensor shape: {samples_ddim.shape}")
        
        # 7.6. Decode the Latent Samples to Image
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        output_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 7.7. Convert Tensor to PIL Image
        output_image = output_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255  # Shape: [H, W, C]
        output_image_pil = Image.fromarray(output_image.astype(np.uint8))
        augmented_images.append(output_image_pil)
    
    return preprocessed_image_pil, augmented_images




# Example usage
if __name__ == '__main__':
    zero123_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Zero123 models
    print('Loading Zero123 models...')
    zero123 = {}
    config_path = './configs/sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config_path)

    model_path = "./105000.ckpt"
    zero123['model'] = load_model_from_config(config, model_path, zero123_device)

    # Carvekit interface for preprocessing
    zero123['carvekit_interface'] = create_carvekit_interface()

    # Define the angles for augmentation
    angles = [
        ("front", 0, 0, 0),
        ("left", 0, -15, 0),
        ("right", 0, 15, 0),
        ("above", -15, 0, 0),
        ("below", 15, 0, 0),
        ("behind", 0, 180, 0),
    ]

    # Load the image (replace with your image path)
    image_path = '/home/pat/diffusion_augmentation/test_images/original.png'
    image = Image.open(image_path)

    # Apply Zero123 augmentations
    preprocessed_image, augmented_images = generate_angles(
        input_image=image,
        angles=angles,
        model=zero123['model'],  # Your loaded model
        carvekit_interface=zero123['carvekit_interface'],  # Your Carvekit interface
        device=zero123_device,  # Your device (e.g., 'cuda:0')
        precision='autocast',  # Use 'autocast' for mixed precision if supported
        h=256,
        w=256,
        ddim_steps=50,
        scale=3.0,
        n_samples=1,
        ddim_eta=1.0
    )

    # Save or visualize the augmented images
    for idx, img in enumerate(augmented_images):
        img.save(f'augmented_view_{angles[idx][0]}.png')

    # Save the preprocessed image
    preprocessed_image.save('preprocessed_image.png')
import torch
from torchvision import transforms, utils
from timm.data import create_dataset
import matplotlib.pyplot as plt

import numpy as np
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import cv2
import einops
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
import os
from torchvision.transforms import InterpolationMode

preprocessor = None

model_name = 'control_v11p_sd15_canny'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

a_prompt = "A single finger print from a scanner surrounded by blank space, only finger, clear image centered, photorealistic"
n_prompt = "multiple, mushed, low quality, cropped, worst quality"
num_samples = 1
image_resolution = 224  # Assuming square images for simplicity
ddim_steps = 30
guess_mode = True
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0
detect_resolution = 224
low_threshold = 1
high_threshold = 255

class AugmentControlDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Store tuples of (image_path, label)
        self.samples = []
        
        for label in os.listdir(img_dir):
            label_path = os.path.join(img_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    self.samples.append((os.path.join(label_path, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def get_image_path(self, idx):
        # Returns the full path of the image at the given index
        return self.samples[idx][0]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_variations(base_dir, image_name):
    """
    Check if there are already three variations of the given image.
    Returns True if three variations exist, False otherwise.
    """
    
    no_ext_name = image_name
    variations = []
    for f in os.listdir(base_dir):
        if f.startswith(no_ext_name):
            variations.append(f)
    return len(variations) >= 3


control_dir = "./control_augmented_images"
def augment(dataset):
    print(type(dataset))

    #how many images are in it 
    print(len(dataset))

    transform = transforms.Compose([
        # Resize the image
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        # Apply center crop if needed (crop to 95% of the image, then resize)
        transforms.CenterCrop(size=int(224 * 0.95)),
        # Convert image to PyTorch tensor
        transforms.ToTensor(),
        # Normalize the image with the specified mean and std deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    count = 0
    for img, label in dataset:
        # Apply the transformations
        # aug_img = transform(img)
        # Display the image in the dataset with matplotlb
        count+=1
        # print(type(img))
        # print(type(label))
        # # print(type(k))
        # print(img)
        # print(label)
        # print(k)
        # Create the directory for the labl
        label_path = os.path.join(control_dir, str(label))
        ensure_dir(label_path)

        # check the directory if it has 3 variations
        check_variations(label_path, f"file{count}")
        # create 3 variations
        results = process("Canny", np.array(img), "", a_prompt, n_prompt, 3, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        # save the original image
        original_filename = f"file{count}_original.png"
        original_path = os.path.join(label_path, original_filename)
        cv2.imwrite(original_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        # save the 3 variations
        for i, result in enumerate(results):
            
            result_filename = f"file{count}_{i}.png"
            result_path = os.path.join(label_path, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        
    additional_dataset = AugmentControlDataset(control_dir, transform=transform)
    print("Control Augmented Dataset: ", len(additional_dataset))
    # Combine datasets
    combined_dataset = ConcatDataset([dataset, additional_dataset])
    print("Combined Dataset: ", len(combined_dataset))
    # Here, you'd return or process the augmented images further as needed
    return combined_dataset



def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor
    ddim_sampler = DDIMSampler(model)
    # Check which preprocessor to use
    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # Save memory
        model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        # Save memory
        model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        # Save memory
        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results
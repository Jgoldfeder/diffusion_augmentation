import torch
from torchvision import transforms, utils
from timm.data import create_dataset
import matplotlib.pyplot as plt
import requests
import numpy as np
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.depth_anything import DepthAnythingDetector

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
from transformers import AutoProcessor, Blip2ForConditionalGeneration,  LlavaForConditionalGeneration
# from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader

import tqdm

preprocessor = None





num_samples = 2
a_prompt = "clear image, photorealistic"
n_prompt = "multiple, mushed, low quality, cropped, worst quality"
image_resolution = 224  # Assuming square images for simplicity
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0
detect_resolution = 256
low_threshold = 50
high_threshold = 200

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
    Check if there are already num_samples and the original image saved variations of the given image.
    Returns True if three variations exist, False otherwise.
    """
    
    no_ext_name = image_name
    variations = []
    ensure_dir(base_dir)
    for f in os.listdir(base_dir):
        if f.startswith(no_ext_name):
            variations.append(f)
    return len(variations) == num_samples + 1

# Function to add the original image if it doesn't exist after checking variations
def add_original_image(base_dir, image_name):
    no_ext_name = image_name
    variations = []
    # original_filename = image_name + "_original.png"
    ensure_dir(base_dir)
    for f in os.listdir(base_dir):
        if f.startswith(no_ext_name):
            variations.append(f)
    if len(variations) == num_samples:
        return True
    return False


# control_dir = "./control_augmented_images"
def augment(dataset, preprocessor = "Canny", control_dir = "./control_augmented_images"):
    if preprocessor == "Canny":
        model_name = 'control_v11p_sd15_canny'
    elif preprocessor == "Depth-Anything":
        model_name = 'control_v11p_sd15_depth_anything'

    control_model = create_model(f'./models/{model_name}.yaml').cpu()
    control_model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda:0'), strict=False)
    control_model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda:0'), strict=False)
    control_model = control_model.cuda()
    ddim_sampler = DDIMSampler(control_model)
    
    # Instantiate the captioning model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    caption_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #choose specific cuda device
    # device = torch.device("cuda:1")
    # caption_model.to(device)
    # print(device)
    # model.to(device)
    print(type(dataset))
    #print the classes in the dataset
    original_dataset = dataset.dataset
    print(original_dataset)
    classes = original_dataset.classes
    #get the original filename of each from the dataset
    # print(dataset.dataset.samples)


    
    transform = transforms.Compose([
        # Resize the image
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        # Apply center crop if needed (crop to 95% of the image, then resize)
        # transforms.CenterCrop(size=int(224 * 0.95)),
        # Convert image to PyTorch tensor
        transforms.ToTensor(),
        # Normalize the image with the specified mean and std deviation
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    count = 0
    batch_size = 10  # Define your batch size based on your hardware capabilities
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in data_loader:
        images, labels = batch
        # check the directory if it has 3 variations then did this batch already
        label_path = os.path.join(control_dir, str(int(labels[0])))
        if check_variations(label_path, f"file{count}"):
            print("SKIPPING BATCH at filename ", label_path, f"file{count}")
            count+=batch_size
            # print("SKIPPING BATCH")
            continue


        raw_images = [to_pil_image(img) for img in images]
        prompts = [
            "USER: <image>\nYou are creating a prompt for a diffusion model in order to recreate this scene of a " + classes[label] + ". Describe the image in detail in as many words as possible, focusing on what colors and where objects are. Note the colors of each object when describing the scene.\nASSISTANT:"
        for label in labels] 
        inputs = processor(prompts, images=raw_images, padding=True, return_tensors="pt").to("cuda")
        # print(prompts)
        with torch.no_grad():
            generated_ids = caption_model.generate(**inputs, max_length=200)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = [caption.split("ASSISTANT: ")[1] if "ASSISTANT: " in caption else caption for caption in captions]
        # print(captions)
        # Process each image-caption pair in the batch
        for img, label, caption in zip(images, labels, captions):
            # Apply the transformations
            # aug_img = transform(img)
            # Display the image in the dataset with matplotlb
            
            # print(type(img))
            # print(type(label))
            # # print(type(k))
            # print(img)
            # print(label)
            # print(k)
            # Create the directory for the labl
            label_path = os.path.join(control_dir, str(int(label)))
            
            #Create the path if it doesn't exist
            ensure_dir(label_path)

            #convert to uint8
            img = (np.array(img) * 255).astype(np.uint8)
            # H, W, C = img.shape``
            # Check if the image is in (Channels, Height, Width) format and transpose it
            if img.shape[2] not in (1, 3, 4):
                # Assuming the input is in (Channels, Height, Width) format
                if img.shape[0] in (1, 3, 4):
                    img = img.transpose(1, 2, 0)  # Convert to (Height, Width, Channels)
                else:
                    raise AssertionError("Image channels must be 1, 3, or 4.")
            

            
            # if add_original_image(label_path, f"file{count}"):
            #     # save the original image
                
            #     original_filename = f"file{count}_original.png"
            #     print("Saving original image for ", str(int(label)), " at ", label_path, " with filename ", original_filename)
            #     original_path = os.path.join(label_path, original_filename)
            #     cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            #     count+=1
            #     continue

            # create variations
            print("Creating variations for ", str(int(label)), " at ", label_path, " with filename ", f"file{count}")
            print(caption)
            results = process(control_model, ddim_sampler, preprocessor, img, caption, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
            
            original_filename = f"file{count}_original.png"
            original_path = os.path.join(label_path, original_filename)
            cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # save the 3 variations
            for i, result in enumerate(results):
                # print(result_path)
                # print(result)
                result_filename = f"file{count}_{i}.png"
                result_path = os.path.join(label_path, result_filename)
                # print(result_path)
                # print("Result:",result.shape)
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            count+=1

        
    additional_dataset = AugmentControlDataset(control_dir, transform=transform)
    print("Control Augmented Dataset: ", len(additional_dataset))
    # Combine datasets
    # combined_dataset = ConcatDataset([dataset, additional_dataset])
    # print("Combined Dataset: ", len(combined_dataset))
    # Here, you'd return or process the augmented images further as needed

    #clean up the models from teh gpu
    del control_model
    del ddim_sampler
    del caption_model
    torch.cuda.empty_cache()
    return additional_dataset


def process(model, ddim_sampler, det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor

    # Check which preprocessor to use
    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()
    
    elif det == 'DepthAnything':
        if not isinstance(preprocessor, DepthAnythingDetector):
            preprocessor = DepthAnythingDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)
        # print("Input image shape: ", input_image.shape)
        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            # print("Detected map shape: ", detected_map.shape)
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
    return results

# If using depth anything to get the depth maps
def depth_anything(img, res:int = 512, colored:bool = True, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_depth_anything
    if model_depth_anything is None:
        model_depth_anything = DepthAnythingDetector(device)
    return remove_pad(model_depth_anything(img, colored=colored)), True

def unload_depth_anything():
    if model_depth_anything is not None:
        model_depth_anything.unload_model()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()
def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)
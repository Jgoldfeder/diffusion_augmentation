import torch
from torchvision import transforms, utils
from timm.data import create_dataset
import matplotlib.pyplot as plt
import requests
import numpy as np
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
# from annotator.depth_anything import DepthAnythingDetector

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

#Libraries for upscaling
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline


from torch.utils.data import Sampler


UPSCALING = False


preprocessor = None

a_prompt = "clear image, photorealistic, real world"
n_prompt = "multiple, mushed, low quality, cropped, worst quality"

ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0

low_threshold = 50
high_threshold = 200

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


def my_collate_fn(batch):
    transform = transforms.Compose([
        transforms.Resize(size=(image_resolution, image_resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    
    original_images = []
    transformed_images = []
    labels = []
    image_names = []
    for img, label, image_name in batch:
        original_images.append(img)  # Store original PIL image
        transformed_images.append(transform(to_pil_image(img)))  # Apply transformation
        image_names.append(image_name)
        labels.append(label)

    return torch.stack(transformed_images), labels, original_images, image_names

class ReverseSampler(Sampler):
    """Sampler that returns data indices in reverse order."""
    
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Return indices from last to first
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self):
        return len(self.data_source)
    
# ['sun_bsegmfmvpxfdgkkg.jpg', 'sun_aubczhogizfeakaf.jpg', 'sun_aveoyjxhnzidqewq.jpg', 'sun_bwlllvsnamiycbfe.jpg', 'sun_agtryqbvjkpiypzo.jpg', 'sun_aufzwmcumasvivyn.jpg', 'sun_brqufrhphspuijou.jpg', 'sun_bhheuiomlapxutgy.jpg', 'sun_adieyvqrqcdifwfg.jpg', 'sun_azvofkmhyuxnlqht.jpg']

# control_dir = "./control_augmented_images"
def augment(dataset, preprocessor = "Canny", control_dir = "./control_augmented_images_test", variations = 15, res = 512, images_per_class = 10, reverse = False):
    global image_resolution
    global num_samples

    image_resolution = res
    
    num_samples = variations

    if preprocessor == "Canny":
        model_name = 'control_v11p_sd15_canny'
    else:
        print("Invalid preprocessor")
        return None

    control_model = create_model(f'./models/{model_name}.yaml').cpu()
    control_model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda:0'), strict=False)
    control_model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda:0'), strict=False)
    control_model = control_model.to('cuda:0')
    ddim_sampler = DDIMSampler(control_model)
    
    # Instantiate the captioning model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    caption_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
   
    original_dataset = dataset.dataset
    classes = original_dataset.classes
    # classes = [
    #     "apples", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottles",
    #     "bowls", "boy", "bridge", "bus", "butterfly", "camel", "cans", "castle", "caterpillar", "cattle",
    #     "chair", "chimpanzee", "clock", "cloud", "cockroach",  "computer keyboard", "couch", "crab", "crocodile", "cups", "dinosaur",
    #     "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
    #     "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    #     "mouse", "mushrooms", "oak_tree", "oranges", "orchid", "otter", "palm_tree", "pears", "pickup_truck", "pine_tree",
    #     "plain", "plates", "poppies", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    #     "roses", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    #     "squirrel", "streetcar", "sunflowes", "sweet_peppers", "table", "tank", "telephone", "television", "tiger", "tractor",
    #     "train", "trout", "tulips", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    # ]
    #get the original filename of each from the dataset
    #print the shape of an image in the dataset
    # print(dataset.dataset[0][0].shape)
    
    # ['110_0079.jpg', '081_0018.jpg', '064_0062.jpg', '148_0113.jpg', '193_0060.jpg', '240_0162.jpg', '141_0048.jpg', '129_0098.jpg', '221_0043.jpg', '105_0251.jpg']
    
    batch_size = 10  # Define your batch size based on your hardware capabilities

    # Start from the back or not
    if reverse:
        reverse_sampler = ReverseSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, sampler=reverse_sampler)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    for batch in data_loader:
        
        print("Variation count: ", variations)
        print("Images per class: ", images_per_class)
        images, labels, original_images, image_names = batch
        print(image_names)
        # print(labels)
        #show the images shape
        # print(images.shape)
        # check the directory if it has 3 variations then did this batch already
        num_digits = 10  # Adjust based on the maximum label number you expect
        formatted_label = f"{labels[0]:0{num_digits}d}"  # This formats the label with leading zeros
        
        label_path = os.path.join(control_dir, formatted_label)
        print("Checking variations for ", label_path, image_names[0].split(".")[0]+"_original.png")
        check_filename = image_names[0].split(".")[0]+"_original.png"

        if os.path.exists(os.path.join(label_path, check_filename)):
            print("SKIPPING BATCH at filename ", label_path, image_names[0].split(".")[0]+"_original.png")
            # count+=batch_size
            # print("SKIPPING BATCH")
            continue


        raw_images = [to_pil_image(img) for img in images]
        # print("Raw images shape: ", raw_images.shape)
        prompts = [
            "USER: <image>\nYou are creating a prompt for a diffusion model in order to recreate this scene of a " + classes[label] + ". Describe the image in detail in as many words as possible, focusing on what colors and where objects are. Note the colors of each object when describing the scene.\nASSISTANT:"
        for label in labels] 
        inputs = processor(prompts, images=raw_images, padding=True, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            generated_ids = caption_model.generate(**inputs, max_length=200)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = [caption.split("ASSISTANT: ")[1] if "ASSISTANT: " in caption else caption for caption in captions]
        # del caption_model
        # del processor
        # torch.cuda.empty_cache()
        # Process each image-caption pair in the batch
        for img, label, caption, og_img, image_name in zip(images, labels, captions, original_images, image_names):
            
            # Create the directory for the labl
            num_digits = 10  # Adjust based on the maximum label number you expect
            formatted_label = f"{label:0{num_digits}d}"  # This formats the label with leading zeros
            label_path = os.path.join(control_dir, formatted_label)
            # label_path = os.path.join(control_dir, str(int(label)))
            
            #Create the path if it doesn't exist
            ensure_dir(label_path)

            # Check if we want to only augment a certain number of images per class
            if images_per_class > 0:
                files_in_dir = os.listdir(label_path)
                print("Files in dir: ", len(files_in_dir)/(variations + 1))
                if len(files_in_dir)/(variations + 1) >= images_per_class:
                    print("Max images per class reached in class ", label)
                    continue
            
            # check if file exists then skip
            print("Original shape: ", og_img.shape)
            original_filename = image_name.split(".")[0]+"_original.png"

            if os.path.exists(os.path.join(label_path, original_filename)):
                print("Original image already exists, ", original_filename,  " - skipping...")
                continue

            #print the image shape
            print(img.shape)
            #convert to uint8
            img = (np.array(img) * 255).astype(np.uint8)

            print("Image shape: ", img.shape)
            # H, W, C = img.shape
            # Check if the image is in (Channels, Height, Width) format and transpose it
            if img.shape[2] not in (1, 3, 4):
                # Assuming the input is in (Channels, Height, Width) format
                if img.shape[0] in (1, 3, 4):
                    img = img.transpose(1, 2, 0)  # Convert to (Height, Width, Channels)
                else:
                    raise AssertionError("Image channels must be 1, 3, or 4.")
        
            

            
            
            # create variations
            print("Creating variations for ", str(int(label)), " at ", label_path, " with filename ", image_name.split(".")[0])
            caption = "Extremely Realistic, Photorealistic, Clear Image, Real World, " + caption
            print(caption)
            results = process(control_model, ddim_sampler, preprocessor, img, caption, a_prompt, n_prompt, num_samples, image_resolution, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
            #print the size of the image about to be saved
            
            
            # save the variations
            for i, result in enumerate(results):
                # print(result_path)
                # print(result)
                result_filename = image_name.split(".")[0]+f"_{i}.png"
                result_path = os.path.join(label_path, result_filename)
                # print(result_path)
                # print("Result:",result.shape)
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            # Save the original image

            original_path = os.path.join(label_path, original_filename)
            og_img = og_img.numpy()

            # Change from (C, H, W) to (H, W, C)
            og_img = np.transpose(og_img, (1, 2, 0))

            # Convert data type to uint8
            og_img = (og_img * 255).astype(np.uint8)
            # print(og_img.shape)
            # print(og_img)
            cv2.imwrite(original_path, cv2.cvtColor(og_img, cv2.COLOR_RGB2BGR))
            

        
    # additional_dataset = AugmentControlDataset(control_dir, transform=transform)
    # print("Control Augmented Dataset: ", len(additional_dataset))
    # Combine datasets
    # combined_dataset = ConcatDataset([dataset, additional_dataset])
    # print("Combined Dataset: ", len(combined_dataset))
    # Here, you'd return or process the augmented images further as needed

    #clean up the models from teh gpu
    del control_model
    del ddim_sampler
    
    torch.cuda.empty_cache()

    return None


def process(model, ddim_sampler, det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor

    # Check which preprocessor to use
    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()
    

    with torch.no_grad():
        input_image = HWC3(input_image)
        # print("Input image shape: ", input_image.shape)
        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image.copy(), detect_resolution), low_threshold, high_threshold)
            # print("Detected map shape: ", detected_map.shape)
            detected_map = HWC3(detected_map)
        
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        print("Detected map shape: ", detected_map.shape)
        control = torch.from_numpy(detected_map.copy()).float().to('cuda:0') / 255.0

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
    return results #+ [detected_map]

# If using depth anything to get the depth maps

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

# if this sript is run check if how many images are in each class in a directory

if __name__ == "__main__":
    label_path="./control_augmented_images_caltech256_512fewshot2"
    # files_in_dir = os.listdir(label_path)
    for label in os.listdir(label_path):
        files_in_dir = os.listdir(os.path.join(label_path, label))
        print("Files in ", label,": ", len(files_in_dir)/16)
        if len(files_in_dir)/16 >= 10:
            print("Max images per class reached in class ", label)
        else:
            
            print("NOT REACHEDDDD")
            break
        
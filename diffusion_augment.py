import torch
from torchvision import transforms, utils
from timm.data import create_dataset
import matplotlib.pyplot as plt
import requests
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
from transformers import AutoProcessor, Blip2ForConditionalGeneration,  LlavaForConditionalGeneration
# from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

preprocessor = None


model_name = 'control_v11p_sd15_canny'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda:0'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda:0'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

variations = 2
a_prompt = "clear image, photorealistic"
n_prompt = "multiple, mushed, low quality, cropped, worst quality"
image_resolution = 224  # Assuming square images for simplicity
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0
detect_resolution = 224
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

    # Instantiate the captioning model
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # caption_model = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    # )
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    caption_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #choose specific cuda device
    # device = torch.device("cuda:1")
    caption_model.to(device)
    print(device)
    # model.to(device)
    print(type(dataset))
    #print the classes in the dataset
    original_dataset = dataset.dataset

    # Check if the original dataset has an attribute named 'classes'
    if hasattr(original_dataset, 'classes'):
        # Print the class names
        print(original_dataset.classes)
    else:
        print("The dataset does not have a 'classes' attribute")
        #how many images are in it 
        print(len(dataset))

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

         # create a caption for the image
        print(type(img))
        raw_image = to_pil_image(img)
        print(type(raw_image))
        # raw_image.show()
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # display(image1)
        prompts = [
            "USER: <image>\nDescribe the image in detail in as many words as possible. Describe the colors, what's in focus, out of focus, on the ground, and in they sky. \nASSISTANT:",
        ]   
        inputs = processor(prompts, images=[raw_image], padding=True, return_tensors="pt").to(device=device, dtype=torch.float16)
        # prompt = "Question: Describe the image in detail in as many words as possible. Answer:"
        # inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)


        generated_ids = caption_model.generate(**inputs)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(caption)

        #convert to uint8
        img = (np.array(img) * 255).astype(np.uint8)
        # H, W, C = img.shape
        # Check if the image is in (Channels, Height, Width) format and transpose it
        if img.shape[2] not in (1, 3, 4):
            # Assuming the input is in (Channels, Height, Width) format
            if img.shape[0] in (1, 3, 4):
                img = img.transpose(1, 2, 0)  # Convert to (Height, Width, Channels)
            else:
                raise AssertionError("Image channels must be 1, 3, or 4.")
        
       
        
        # create variations
        results = process("Canny", img, caption, a_prompt, n_prompt, variations, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        # save the original image
        original_filename = f"file{count}_original.png"
        original_path = os.path.join(label_path, original_filename)
        # print("Original:",img.shape)
        cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # save the 3 variations
        for i, result in enumerate(results):
            
            result_filename = f"file{count}_{i}.png"
            result_path = os.path.join(label_path, result_filename)
            # print("Result:",result.shape)
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
        print("Input image shape: ", input_image.shape)
        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            print("Detected map shape: ", detected_map.shape)
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
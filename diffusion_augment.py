import os
import torch
import numpy as np
import cv2
import einops
import random

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torchvision import transforms
from pytorch_lightning import seed_everything
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import BitsAndBytesConfig
from diffusers import StableDiffusionPipeline


# Constants
UPSCALING = False

# ControlNet+-
A_PROMPT = "clear image, photorealistic, real world"
N_PROMPT = "multiple, mushed, low quality, cropped, worst quality"
DDIM_STEPS = 20
GUESS_MODE = False
STRENGTH = 1.0
SCALE = 7.5
SEED = -1  # Use -1 for random seeds
ETA = 0.0

# Canny thresholds
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 200


DEFAULT_RES = 512
DEFAULT_VARIATIONS = 15
DEFAULT_SHOTS = 0


CAPTION_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


BATCH_SIZE = 10
NUM_DIGITS = 10


LLAMA_PROMPT = "USER: <image>\nYou are creating a prompt for a diffusion model \
                in order to recreate this scene of a {}. Describe the image \
                in detail in as many words as possible, focusing on what colors \
                and where objects are. Note the colors of each object when \
                describing the scene.\nASSISTANT:"
MAX_CAPTION_LENGTH = 200


def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def check_variations(base_dir, image_name):
    """
    Check if there are already num_samples and the original image saved variations of the given image.
    Returns True if three variations exist, False otherwise.
    """
    
    no_ext_name = image_name
    variations = []
    ensure_dir(base_dir)
    for f in os.listdir(base_dir):
        if f.startswith(no_ext_name): variations.append(f)

    # return len(variations) == num_samples + 1


class ReverseSampler(Sampler):
    """Sampler that returns data indices in reverse order."""
    
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Return indices from last to first
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self):
        return len(self.data_source)
    

def augment(
    dataset, 
    preprocessor="Canny", 
    control_dir="./control_augmented_images_test", 
    variations=DEFAULT_VARIATIONS, 
    res=DEFAULT_RES, 
    images_per_class=DEFAULT_SHOTS, 
    classes=None,
    bad_aug=False,
    ):


    def my_collate_fn(batch):
        transform = transforms.Compose([
            transforms.Resize(
                size=(res, res), 
                interpolation=InterpolationMode.BICUBIC
                ),
            transforms.ToTensor(),
        ])
        
        original_images = []
        transformed_images = []
        labels = []

        for img, label in batch:
            img = transforms.ToTensor()(img)
            # If the image is grayscale, add a channel dimension
            if img.ndim == 2: img = img.unsqueeze(0)

            # If the image has a single channel
            # Convert grayscale to RGB by repeating channels
            elif img.shape[0] == 1: img = img.repeat(3, 1, 1)

            # Store original and transformed images
            original_images.append(img)
            transformed_images.append(transform(to_pil_image(img)))
            labels.append(label)

        return torch.stack(transformed_images), labels, original_images
    

    # Preprocessor
    if preprocessor == "Canny": model_name = "control_v11p_sd15_canny"
    else:
        print("Invalid preprocessor")
        return None
    

    # Captioning Model
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    # Instantiate the captioning model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16
    )
    caption_model = LlavaForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_ID, 
        quantization_config=quantization_config, 
        device_map="auto"
        )
    

    # ControlNet Model
    if not bad_aug:
        control_model = create_model(f'./models/{ model_name }.yaml').cpu()
        control_model.load_state_dict(
            load_state_dict(
                "./models/v1-5-pruned.ckpt",  
                location='cuda:0'
                ),
            strict=False
            )
        control_model.load_state_dict(
            load_state_dict(
                f"./models/{ model_name }.pth", 
                location='cuda:0'
                ), 
            strict=False
            )
        control_model = control_model.to('cuda:0')
        ddim_sampler = DDIMSampler(control_model)

    # Stable Diffusion Model
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
            ).to("cuda")
        


    # original_dataset = dataset.dataset  
    # classes = original_dataset.classes

    #original_dataset = dataset
    #classes = sorted(
    #     list(dataset.class_to_idx.keys()), 
    #     key=lambda x: dataset.class_to_idx[x]
    #     )

    # classes = dataset.classes
    # classes = ["am general hummer suv 2000", "acura rl sedan 2012", "acura tl sedan 2012", "acura tl type-s 2008", "acura tsx sedan 2012", "acura integra type r 2001", "acura zdx hatchback 2012", "aston martin v8 vantage convertible 2012", "aston martin v8 vantage coupe 2012", "aston martin virage convertible 2012", "aston martin virage coupe 2012", "audi rs 4 convertible 2008", "audi a5 coupe 2012", "audi tts coupe 2012", "audi r8 coupe 2012", "audi v8 sedan 1994", "audi 100 sedan 1994", "audi 100 wagon 1994", "audi tt hatchback 2011", "audi s6 sedan 2011", "audi s5 convertible 2012", "audi s5 coupe 2012", "audi s4 sedan 2012", "audi s4 sedan 2007", "audi tt rs coupe 2012", "bmw activehybrid 5 sedan 2012", "bmw 1 series convertible 2012", "bmw 1 series coupe 2012", "bmw 3 series sedan 2012", "bmw 3 series wagon 2012", "bmw 6 series convertible 2007", "bmw x5 suv 2007", "bmw x6 suv 2012", "bmw m3 coupe 2012", "bmw m5 sedan 2010", "bmw m6 convertible 2010", "bmw x3 suv 2012", "bmw z4 convertible 2012", "bentley continental supersports conv. convertible 2012", "bentley arnage sedan 2009", "bentley mulsanne sedan 2011", "bentley continental gt coupe 2012", "bentley continental gt coupe 2007", "bentley continental flying spur sedan 2007", "bugatti veyron 16.4 convertible 2009", "bugatti veyron 16.4 coupe 2009", "buick regal gs 2012", "buick rainier suv 2007", "buick verano sedan 2012", "buick enclave suv 2012", "cadillac cts-v sedan 2012", "cadillac srx suv 2012", "cadillac escalade ext crew cab 2007", "chevrolet silverado 1500 hybrid crew cab 2012", "chevrolet corvette convertible 2012", "chevrolet corvette zr1 2012", "chevrolet corvette ron fellows edition z06 2007", "chevrolet traverse suv 2012", "chevrolet camaro convertible 2012", "chevrolet hhr ss 2010", "chevrolet impala sedan 2007", "chevrolet tahoe hybrid suv 2012", "chevrolet sonic sedan 2012", "chevrolet express cargo van 2007", "chevrolet avalanche crew cab 2012", "chevrolet cobalt ss 2010", "chevrolet malibu hybrid sedan 2010", "chevrolet trailblazer ss 2009", "chevrolet silverado 2500hd regular cab 2012", "chevrolet silverado 1500 classic extended cab 2007", "chevrolet express van 2007", "chevrolet monte carlo coupe 2007", "chevrolet malibu sedan 2007", "chevrolet silverado 1500 extended cab 2012", "chevrolet silverado 1500 regular cab 2012", "chrysler aspen suv 2009", "chrysler sebring convertible 2010", "chrysler town and country minivan 2012", "chrysler 300 srt-8 2010", "chrysler crossfire convertible 2008", "chrysler pt cruiser convertible 2008", "daewoo nubira wagon 2002", "dodge caliber wagon 2012", "dodge caliber wagon 2007", "dodge caravan minivan 1997", "dodge ram pickup 3500 crew cab 2010", "dodge ram pickup 3500 quad cab 2009", "dodge sprinter cargo van 2009", "dodge journey suv 2012", "dodge dakota crew cab 2010", "dodge dakota club cab 2007", "dodge magnum wagon 2008", "dodge challenger srt8 2011", "dodge durango suv 2012", "dodge durango suv 2007", "dodge charger sedan 2012", "dodge charger srt-8 2009", "eagle talon hatchback 1998", "fiat 500 abarth 2012", "fiat 500 convertible 2012", "ferrari ff coupe 2012", "ferrari california convertible 2012", "ferrari 458 italia convertible 2012", "ferrari 458 italia coupe 2012", "fisker karma sedan 2012", "ford f-450 super duty crew cab 2012", "ford mustang convertible 2007", "ford freestar minivan 2007", "ford expedition el suv 2009", "ford edge suv 2012", "ford ranger supercab 2011", "ford gt coupe 2006", "ford f-150 regular cab 2012", "ford f-150 regular cab 2007", "ford focus sedan 2007", "ford e-series wagon van 2012", "ford fiesta sedan 2012", "gmc terrain suv 2012", "gmc savana van 2012", "gmc yukon hybrid suv 2012", "gmc acadia suv 2012", "gmc canyon extended cab 2012", "geo metro convertible 1993", "hummer h3t crew cab 2010", "hummer h2 sut crew cab 2009", "honda odyssey minivan 2012", "honda odyssey minivan 2007", "honda accord coupe 2012", "honda accord sedan 2012", "hyundai veloster hatchback 2012", "hyundai santa fe suv 2012", "hyundai tucson suv 2012", "hyundai veracruz suv 2012", "hyundai sonata hybrid sedan 2012", "hyundai elantra sedan 2007", "hyundai accent sedan 2012", "hyundai genesis sedan 2012", "hyundai sonata sedan 2012", "hyundai elantra touring hatchback 2012", "hyundai azera sedan 2012", "infiniti g coupe ipl 2012", "infiniti qx56 suv 2011", "isuzu ascender suv 2008", "jaguar xk xkr 2012", "jeep patriot suv 2012", "jeep wrangler suv 2012", "jeep liberty suv 2012", "jeep grand cherokee suv 2012", "jeep compass suv 2012", "lamborghini reventon coupe 2008", "lamborghini aventador coupe 2012", "lamborghini gallardo lp 570-4 superleggera 2012", "lamborghini diablo coupe 2001", "land rover range rover suv 2012", "land rover lr2 suv 2012", "lincoln town car sedan 2011", "mini cooper roadster convertible 2012", "maybach landaulet convertible 2012", "mazda tribute suv 2011", "mclaren mp4-12c coupe 2012", "mercedes-benz 300-class convertible 1993", "mercedes-benz c-class sedan 2012", "mercedes-benz sl-class coupe 2009", "mercedes-benz e-class sedan 2012", "mercedes-benz s-class sedan 2012", "mercedes-benz sprinter van 2012", "mitsubishi lancer sedan 2012", "nissan leaf hatchback 2012", "nissan nv passenger van 2012", "nissan juke hatchback 2012", "nissan 240sx coupe 1998", "plymouth neon coupe 1999", "porsche panamera sedan 2012", "ram c/v cargo van minivan 2012", "rolls-royce phantom drophead coupe convertible 2012", "rolls-royce ghost sedan 2012", "rolls-royce phantom sedan 2012", "scion xd hatchback 2012", "spyker c8 convertible 2009", "spyker c8 coupe 2009", "suzuki aerio sedan 2007", "suzuki kizashi sedan 2012", "suzuki sx4 hatchback 2012", "suzuki sx4 sedan 2012", "tesla model s sedan 2012", "toyota sequoia suv 2012", "toyota camry sedan 2012", "toyota corolla sedan 2012", "toyota 4runner suv 2012", "volkswagen golf hatchback 2012", "volkswagen golf hatchback 1991", "volkswagen beetle hatchback 2012", "volvo c30 hatchback 2012", "volvo 240 sedan 1993", "volvo xc90 suv 2007", "smart fortwo convertible 2012"]
    

    # Load the classes


    # print(classes)
    # return
    # i think for cifar?
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

    #flowers
    # classes = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
    
    # ['110_0079.jpg', '081_0018.jpg', '064_0062.jpg', '148_0113.jpg', '193_0060.jpg', '240_0162.jpg', '141_0048.jpg', '129_0098.jpg', '221_0043.jpg', '105_0251.jpg']


    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=my_collate_fn
        )


    count = 0
    for batch in data_loader:
        
        print(f"Variation count: { variations }")
        print(f"Images per class: { images_per_class }", images_per_class)

        images, labels, original_images = batch
        # This formats the label with leading zeros
        formatted_label = f"{labels[0]:0{NUM_DIGITS}d}"
        label_path = os.path.join(control_dir, formatted_label)

        raw_images = [to_pil_image(img) for img in images]
        prompts = [LLAMA_PROMPT.format(label) for label in labels] 
        inputs = processor(prompts, images=raw_images, padding=True, return_tensors="pt").to("cuda:0")


        # Generate captions
        with torch.no_grad():
            generated_ids = caption_model.generate(**inputs, max_length=200)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = [caption.split("ASSISTANT: ")[1] if "ASSISTANT: " in caption else caption for caption in captions]
    

        for img, label, caption, og_img in zip(images, labels, captions, original_images):
            
            # Create the directory for the label
            formatted_label = f"{label:0{NUM_DIGITS}d}"  # This formats the label with leading zeros
            label_path = os.path.join(control_dir, formatted_label)
            ensure_dir(label_path)


            # Check if we want to only augment a certain number of images per class
            files_in_dir = os.listdir(label_path)
            print(f"Files in dir: { len(files_in_dir) / (variations + 1) }")
            if len(files_in_dir) / (variations + 1) == images_per_class: continue


            #convert to uint8
            img = (np.array(img) * 255).astype(np.uint8)

            # H, W, C = img.shape
            # Check if the image is in (Channels, Height, Width) format and transpose it
            if img.shape[2] not in (1, 3, 4):
                # Assuming the input is in (Channels, Height, Width) format
                # Convert to (Height, Width, Channels)
                if img.shape[0] in (1, 3, 4): img = img.transpose(1, 2, 0)
                else: raise AssertionError("Image channels must be 1, 3, or 4.")

            # Create variations
            print(f"Creating variations for { int(label) } at { label_path } with filename file{ count }")
            caption = f"Extremely Realistic, Photorealistic, Clear Image, Real World, { classes[label]} .{ caption }"
            print(caption)

            if bad_aug: results = [np.array(el) for el in pipe([caption] * variations).images]
            else:
                results = process(
                    control_model, 
                    ddim_sampler, 
                    preprocessor, 
                    img, 
                    caption, 
                    variations, 
                    res, 
                    res,
                    )
                             
            # Save the variations
            for i, result in enumerate(results):
                result_filename = f"file{ count }"+f"_{ i }.png"
                result_path = os.path.join(label_path, result_filename)
                print("respath", result_path)
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
            # Save the original image
            original_filename = f"file{count}_original.png"
            if os.path.exists(os.path.join(label_path, original_filename)):
                print(f"Original image already exists, { original_filename } - skipping...")
                continue
            original_path = os.path.join(label_path, original_filename)
            og_img = og_img.numpy()
            og_img = np.transpose(og_img, (1, 2, 0)) # Change from (C, H, W) to (H, W, C)
            og_img = (og_img * 255).astype(np.uint8) # Convert data type to uint8
            cv2.imwrite(original_path, cv2.cvtColor(og_img, cv2.COLOR_RGB2BGR))
            count += 1
            
    # Cleanup
    del control_model
    del ddim_sampler
    torch.cuda.empty_cache()

    return None


# This is ControlNet
def process(
        model, 
        ddim_sampler, 
        det, 
        input_image, 
        prompt, 
        num_samples, 
        image_resolution, 
        detect_resolution,
        a_prompt=A_PROMPT, 
        n_prompt=N_PROMPT,  
        ddim_steps=DDIM_STEPS, 
        guess_mode=GUESS_MODE, 
        strength=STRENGTH, 
        scale=SCALE, 
        seed=SEED, 
        eta=ETA, 
        low_threshold=LOW_THRESHOLD, 
        high_threshold=HIGH_THRESHOLD
        ):
    global preprocessor

    # Check which preprocessor to use
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
        
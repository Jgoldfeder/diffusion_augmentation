import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def upscale_and_save_image(image, label, pipeline, output_dir, count, classes):
    # Convert tensor image to PIL for processing if needed
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    num_digits = 10  # Adjust based on the maximum label number you expect
    formatted_label = f"{label:0{num_digits}d}"  # This formats the label with leading zeros
    label_dir = os.path.join(output_dir, formatted_label)
    
    #classes of cifar 100
    cifar100_classes = [
        "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
        "kangaroo", "keyboard", "lamp", "lawn-mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse", "mushroom",
        "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck",
        "pine tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
        "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
        "sunflower", "sweet pepper", "table", "tank", "telephone", "television",
        "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
        "willow tree", "wolf", "woman", "worm"
    ]
    print(len(cifar100_classes))
    # Upscale the image x1
    prompt = f"Extremely Realistic, Photorealistic, Clear Image, Real World, {cifar100_classes[label]}"
    print(prompt)
    upscaled_image = pipeline(prompt=prompt, image=image).images[0]
    ensure_dir(label_dir)
    output_path = os.path.join(label_dir, f'upscaled_x1_{count}.png')  # Consider generating unique names
    upscaled_image.save(output_path)

    # Upscale the image x2
    # upscaled_image = pipeline(prompt=prompt, image=upscaled_image).images[0]
    # output_path = os.path.join(label_dir, f'upscaled_x2_{count}.png')  # Consider generating unique names

    upscaled_image.save(output_path)
    print(f"Saved upscaled image to {output_path}")

def upscale(dataset, output_dir = "./upscale_test"):
    # Load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    # print(classes)
    # Use DataLoader to handle batching
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # count = 0
    # Process each batch
    for i, (image, label) in enumerate(dataset):
        upscale_and_save_image(image, label, pipeline, output_dir, i, dataset.classes)

if __name__ == "__main__":
    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Set the path to your CIFAR-100 training data
    train_dir = "./cifar100-128/train"

    # Create the dataset
    train_dataset = ImageFolder(train_dir, transform=transform)

    # Define the output directory where upscaled images will be saved
    output_dir = "./cifar100-512"

    # Upscale all images in the CIFAR-100 training dataset
    upscale(train_dataset, output_dir=output_dir)
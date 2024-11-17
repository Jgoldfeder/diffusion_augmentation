from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch

# Load the ControlNet model specific for color preservation
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-color", torch_dtype=torch.float16)

# Load the Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# Enable advanced memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Use UniPCMultistepScheduler for optimized inference
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Function to resize and process the input image
def preprocess_image_for_color_control(image_path):
    """
    Preprocess an image for use with the color-preserving ControlNet model.
    
    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - image: Processed image (PIL.Image).
    """
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    return img

# Input image path
image_path = "torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg"
processed_image = preprocess_image_for_color_control(image_path)

# Stable Diffusion prompt
prompt = ["ak47, best quality, extremely detailed"]
negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"]

# Generator for reproducibility
generator = [torch.Generator(device="cpu").manual_seed(2) for _ in range(len(prompt))]

# Run the pipeline
output = pipe(
    prompt=prompt,
    image=processed_image,
    negative_prompt=negative_prompt * len(prompt),
    generator=generator,
    num_inference_steps=20,
)

# Save and display the output images
for i, img in enumerate(output.images):
    img.save(f"output_color_preserved_image_{i}.png")
    img.show()

print("Color-preserved images saved and displayed.")
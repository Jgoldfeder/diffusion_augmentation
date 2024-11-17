from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import torch
from controlnet_aux import MidasDetector, CannyDetector, SamDetector

# Load the preprocessors
canny = CannyDetector()
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

def preprocess_image(image_path, preprocessor_type):
    """
    Processes an image with the specified preprocessor type.

    Parameters:
    - image_path (str): Path to the input image.
    - preprocessor_type (str): Type of preprocessor ('canny', 'midas', 'segmentation').

    Returns:
    - processed_image: The processed image (PIL.Image).
    """
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    
    if preprocessor_type == "canny":
        # Canny Edge Detection
        np_img = np.array(img)
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(np_img, low_threshold, high_threshold)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        processed_image = Image.fromarray(edges)
    elif preprocessor_type == "midas":
        # Midas Depth Estimation
        processed_image = midas(img)
    elif preprocessor_type == "segmentation":
        # Segmentation using SAM
        processed_image = sam(img)
    else:
        raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")
    
    return processed_image

# Example usage:
image_path = "torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg"
preprocessor_type = "segmentation"  # Options: "canny", "midas", "segmentation"

processed_image = preprocess_image(image_path, preprocessor_type)
processed_image.show()  # Display the image
processed_image.save(f"output_{preprocessor_type}.png")  # Save the processed image


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

pipe.enable_xformers_memory_efficient_attention()

prompt = ["ak47, best quality, extremely detailed"]
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

output = pipe(
    prompt,
    processed_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
    generator=generator,
    num_inference_steps=20,
)

# Save each image to the current directory
for i, img in enumerate(output.images):
    img.save(f"output_image_{i}.png")
print("Images saved to the current directory.")
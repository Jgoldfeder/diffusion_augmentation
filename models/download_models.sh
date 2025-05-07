#!/bin/bash

# NOTE make sure inside the models folder when running

# Download files for Control Net
echo "Downloading Control Net files..."
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O v1-5-pruned.ckpt
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth -O control_v11p_sd15_canny.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth -O control_v11f1p_sd15_depth.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth -O control_v11p_sd15_seg.pth

# Download files for Color Control Net
echo "Downloading Color Control Net files..."
# wget https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth
wget https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=10r_u7nSi2v5yQR1_EyeN4X79JmeLxTAZ&export=download&confirm=t&uuid=79430cd7-b609-4ba6-a778-fec750cafbd0" -O color_img2img_palette.pt
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth -O sk_model.pth
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth -O sk_model2.pth

# Download files for Zero123
echo "Downloading Zero123 files..."
wget https://huggingface.co/spaces/cvlab/zero123-live/resolve/d7776c37857ae042bf9e31f74d54f736a432ed8a/105000.ckpt -O 105000.ckpt

# Download files for runwayml
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./runwayml-stable-diffusion-v1-5 --local-dir-use-symlinks False

echo "All files downloaded successfully."
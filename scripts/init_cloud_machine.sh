#!/bin/bash
# NOTE after running all commands need to wandb login

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/Jgoldfeder/diffusion_augmentation.git
cd diffusion_augmentation
git checkout learnable_pipeline

conda create --name diffaug python=3.10.15 -y
conda init
source ~/.bashrc
conda activate diffaug
conda install pip

git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/

pip install -r requirements.txt

# note this download step failed once, then when trying again took less than five mins
cd models
chmod +x ./download_models.sh
while true; do
    timeout 10m ./download_models.sh && break
    echo "Download failed or timed out. Retrying..."
    sleep 5  # Optional: Wait 5 seconds before retrying
done
cd ..

# need this line so that cuda memory doesn't run out so quick
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> ~/.bashrc
source ~/.bashrc

# next we download the datasets
mkdir torch
cd torch
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1
tar -xvf '256_ObjectCategories.tar?download=1' 
mkdir caltech256
mv 256_ObjectCategories caltech256
wget https://www.kaggle.com/api/v1/datasets/download/rickyyyyyyy/torchvision-stanford-cars
unzip torchvision-stanford-cars
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf images.tar
mkdir stanford_dogs
mv Images stanford_dogs/Images
wget https://www.kaggle.com/api/v1/datasets/download/waseemalastal/the-oxford-flowers-102-dataset
unzip the-oxford-flowers-102-dataset
mv flower_data flowers102
curl -L -o ./food-101.zip  https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101
unzip ./food-101.zip -d ./food-101
cd ..

chmod +x ./scripts/create_fewshot_datasets.sh
./scripts/create_fewshot_datasets.sh

# TODO wandb login here
python
import wandb
wandb.login()
# authenticate in browser

# TODO For controlnet (DO we need to do this?) I think no
# probably use sed
# * ~/miniconda3/envs/diffaug/lib/python3.10/site-packages/basicsr/data/degradations.py
# * Change the import line from functional_tensor to just functional

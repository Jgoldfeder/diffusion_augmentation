mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
# TODO - add to .bashrc the following line: export PATH="/home/<username>/miniconda3/bin:$PATH"

git clone https://github.com/Jgoldfeder/diffusion_augmentation.git
cd diffusion_augmentation
git checkout learnable_pipelin

conda create --name diffaug python=3.10.15
conda activate diffaug
conda install pip

git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/

pip install -r requirements.txt

cd models
chmod +x ./download_models.sh
./download_models.sh
cd ..

# TODO For controlnet
# probably use sed
* ~/miniconda3/envs/diffaug/lib/python3.10/site-packages/basicsr/data/degradations.py
* Change the import line from functional_tensor to just functional
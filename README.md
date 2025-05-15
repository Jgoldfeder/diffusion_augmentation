# One Image, Many Views: Evolving Task-Specific Augmentations for One-Shot Classification
## Vaibhav Sourirajan, Sean Zhang

### Setup
- install conda
    - mkdir -p ~/miniconda3
    - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    - bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    - rm ~/miniconda3/miniconda.sh
- add to .bashrc the following line: export PATH="/home/<username>/miniconda3/bin:$PATH"
- create environment (conda create —name diffaug python=3.10.15)
- `conda install pip`
- `conda activate diffaug`
- run the comments at the top of requirements.txt first
- `pip install -r requirements.txt`
- Set up conda environment with jupyter kernel
    - `conda install -c conda-forge ipykernel`
    - `python -m ipykernel install --user --name=diffaug`
    - install jupyter extension in VS Code
- `./models/download_models.sh` to install the model weights and checkpoints
- `huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/runwayml-stable-diffusion-v1-5 --local-dir-use-symlinks False` to install runwayml stable diffusion v1.5 for inference without API calls

### How to Run
The primary script for running the genetic algorithm is `genetic_algorithm.py`. We use the following flags to configure the experiment:

#### Genetic Algorithm Parameters
- `num_generations`: Number of generations to run the genetic algorithm (default: 50)
- `sol_per_pop`: Number of solutions (augmentation strategies) in each generation (default: 20)
- `num_parents_mating`: Number of parents selected for mating in each generation (default: 8)
- `keep_elitism`: Number of best solutions to keep from previous generation (default: 2)
- `keep_parents`: Number of parents to keep from previous generation (default: 2)
- `mutation_percent`: Percentage of solutions to mutate in each generation (default: 10)

#### Dataset Parameters
- `dataset`: Dataset to run on (options: 'cifar10', 'cifar100', 'mini_imagenet')
- `num_ways`: Number of classes in the few-shot learning task (default: 5)
- `num_shots`: Number of support examples per class (default: 5)
- `subset`: Subset of the dataset to run on (e.g., 'train', 'val', 'test')
- `seed`: Random seed for reproducibility (default: 42)

#### Model Parameters
- `model_type`: Downstream classification model (options: 'resnet18', 'resnet50', 'conv4')
- `tree_depth`: Depth of augmentation tree (default: 3)
- `num_augmentations_per_image`: Number of augmentations to apply per image (default: 1)
- `num_iterations_for_val`: Number of iterations for validation (few-shot case)
- `num_iterations_for_test`: Number of iterations for testing

#### Clustering Parameters
- `one_shot_clustering`: Whether to use one-shot clustering (default: False)
- `clustering_model_name`: Name of model to encode images for clustering (eg. resnet50, vit)

### Example Usage
```bash
python genetic_algorithm.py \
    --num_generations 50 \
    --sol_per_pop 20 \
    --num_parents_mating 8 \
    --dataset cifar10 \
    --num_ways 5 \
    --num_shots 5 \
    --model_type resnet18 \
    --tree_depth 3 \
    --seed 42
```

### Requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- scikit-learn
- tqdm

### Installation
```bash
pip install -r requirements.txt
```

### Project Structure
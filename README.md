# Learning Task-Specific Image Augmentations with Generative Models
## Judah Goldfeder, Vaibhav Sourirajan, Shreyes Kaliyur, Patrick Puma, Hod Lipson

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
- num_generations: number of generations to run
- sol_per_pop: number of solutions in each generation
- num_parents_mating: number of parents to mate
- keep_elitism: number of best solutions to keep from previous generation
- keep_parents: number of parents to keep from previous generation
- mutation_percent: percentage of solutions to mutate
- dataset: dataset to run on
- num_ways: number of ways in the few shot dataset
- num_shots: number of shots in the few shot dataset
- subset: subset of the dataset to run on

Sample Command:
```
$ nohup python genetic_algorithm.py --num_generations 10 --sol_per_pop 14 --num_parents_mating 6 --keep_elitism 1 --keep_parents 1 --mutation_percent 10 --dataset caltech256 --num_ways 5 --num_shots 2 --subset 42 --model_type resnet50 --tree_depth 3 --num_augmentations_per_image 5 --num_iterations_for_val 20 --num_iterations_for_test 200 --seed 42 &
```
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
- Follow the directions in `scripts/init_cloud_machine.sh` to download the specific datasets required (primarily for non torchvision datasets)
- `./scripts/create_fewshot_datasets.sh`

### How to Run Genetic Algorithm
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
- model_type: downstream classification model
- tree_depth: depth of the augmentation tree
- num_augmentations_per_image: number of augmented images to generate per original image
- num_iterations_for_val: number of training iterations for validation (few-shot fitness function case)
- num_iterations_for_test: number of training iterations for final testing
- seed: random seed for reproducibility
- one_shot_training_loss: (optional) whether to use one shot training loss
- one_shot_clustering: (optional) whether to use one shot clustering
- clustering_model_name: image encoder used in clustering fitness function

Sample Command:
```
$ nohup python genetic_algorithm.py --num_generations 10 --sol_per_pop 14 --num_parents_mating 6 --keep_elitism 1 --keep_parents 1 --mutation_percent 10 --dataset caltech256 --num_ways 5 --num_shots 2 --subset 42 --model_type resnet50 --tree_depth 3 --num_augmentations_per_image 5 --num_iterations_for_val 20 --num_iterations_for_test 200 --seed 42 &
```

For visualization, you can use `clustering_visualization.py` with the following parameters:

Required Parameters:
- dataset: dataset to run on
- num_ways: number of ways in the few shot dataset
- num_shots: number of shots in the few shot dataset
- subset: subset of the dataset to run on
- model_type: downstream classification model
- tree_depth: depth of the augmentation tree
- num_augmentations_per_image: number of augmented images to generate per original image
- seed: random seed for reproducibility
- genome: specific augmentation genome to visualize (comma-separated list of numbers)

Visualization-specific Parameters:
- image_encoder: which image encoder to use for feature extraction (options: 'vit224', 'vit', 'resnet50')
- dimension_reduction: which dimension reduction method to use (options: 'umap', 'tsne', 'umap3d', 'tsne3d')

Sample Command:
```
$ nohup python clustering_visualization.py --dataset caltech256 --num_ways 5 --num_shots 2 --subset 43 --model_type resnet50 --tree_depth 3 --num_augmentations_per_image 10 --seed 42 --dimension_reduction umap --genome 6,0.3025995038712244,1,0.3,6,0.33170061062325357 --image_encoder resnet50 &
```

Note: The visualization script will save the plots in the `clustering_visualizations` directory with filenames following the pattern: `{dataset_name}_{image_encoder}_{dimension_reduction}_projection.png`

### How to Replicate Experiments
You can view the spreadsheet of all results and which subsets they were ran on can be found [here](https://docs.google.com/spreadsheets/d/17kz35sNkv3Tmv37GMa536A67m0ksgJCVSKknbhRWmfo/edit?usp=sharing). 

For any experiment:
- We create the subset mentioned in the spreadsheet by modifying the `scripts/create_fewshot_datasets.sh` script.
- We run a genetic algorithm with the appropriate parameters which logs the best tree and fitness to wandb. Note all experiments were done using an RTX 4090
- We execute `test_tree_accuracy_new.py` with the corresponding genome and experiment params (5 way, 1 shot, dataset) which will log final results to wandb. We take the highest validation accuracy and average across all trials.s
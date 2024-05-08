
# 2 variations 1 shot 5 ways
CUDA_VISIBLE_DEVICES=0 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-5" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512   --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 8 --variations 2 --diffaug-fewshot=1 --classes 0 1 2 3 4


CUDA_VISIBLE_DEVICES=1 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-5" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 24  --diffaug-fewshot=1 --classes 0 1 2 3 4


# 2 variations 1 shot 10 ways
CUDA_VISIBLE_DEVICES=2 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-10" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 8 --variations 2 --diffaug-fewshot=1 --classes 0 1 2 3 4 5 6 7 8 9


CUDA_VISIBLE_DEVICES=3 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-10" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512     --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 24  --diffaug-fewshot=1 --classes 0 1 2 3 4 5 6 7 8 9


# 2 variations 1 shot all ways
CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-all" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512  --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 8 --variations 2 --diffaug-fewshot=1 


CUDA_VISIBLE_DEVICES=5 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-all" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 24  --diffaug-fewshot=1  














# 10 variations 1 shot 5 ways
CUDA_VISIBLE_DEVICES=6 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-5" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 2 --variations 10 --diffaug-fewshot=1 --classes 0 1 2 3 4


CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-5" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 22  --diffaug-fewshot=1 --classes 0 1 2 3 4


# 10 variations 1 shot 10 ways
CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-10" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512     --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 2 --variations 10 --diffaug-fewshot=1 --classes 0 1 2 3 4 5 6 7 8 9


CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-10" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512     --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 22  --diffaug-fewshot=1 --classes 0 1 2 3 4 5 6 7 8 9


# 10 variations 1 shot all ways
CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-all" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "exp" --recipe "sgd-pretrain-fullaug"  --repeats 2 --variations 10 --diffaug-fewshot=1 


CUDA_VISIBLE_DEVICES=4 python3 train.py /data/torch/caltech256/caltech256 --dataset torch/caltech256  --model=resnet50  --num-classes=257  --log-wandb --experiment "caltech256 1shot 2-1-all" --diffaug-dir=/data/puma_envs/control_augmented_images_caltech256_512    --name "base" --recipe "sgd-pretrain-fullaug"  --repeats 22  --diffaug-fewshot=1  

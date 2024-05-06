# Diffusion Agumentation
## Judah Goldfeder, Gabe Guo, Gabriel Trigo, Patrick Puma, Hod Lipson

### Requirments
conda env create -f environment.yaml --prefix=/data/puma_envs/diffusion_augment

conda activate diffusion_augment


### Example script

CUDA_VISIBLE_DEVICES=2 python train.py torch/sun397 --dataset torch/sun397 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-no-variations" --diffaug-dir=./control_augmented_images_depth --diffaug-fewshot=20 --variations --


CUDA_VISIBLE_DEVICES=0 python train.py torch/caltech256 --dataset torch/caltech256 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=256 --opt=sgd --weight-decay=1e-4 --dataset-download --diffaug --diffaug-fewshot=25 --diffaug-dir=./control_augmented_images_caltech256_512fewshot --diffaug-resolutions=512

CUDA_VISIBLE_DEVICES=0 python train.py torch/sun397 --dataset torch/sun397 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --dataset-download --diffaug --diffaug-fewshot=25 --diffaug-dir=./control_augmented_images_sun397_512fewshot --diffaug-resolutions=512

### Sun397 with augmentation

## First Create the dataset
CUDA_VISIBLE_DEVICES=0 python train.py torch/caltech256 --dataset torch/caltech256 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=256 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" --diffaug --diffaug-dir=./control_augmented_images_caltech256

--diffaug is a flag that tells the model to generate diffusion augmentations. This will create the augmented images and store them in the directory specified by --diffaug-dir. Once the dataset is created, the images will be loaded in from this directory and the program will end. Do not set this if we want to train a model

--diffaug-dir=./control_augmented_images_depth is the directory where the augmented images are stored. This is the directory that is created when the dataset is created.

## Then train with the dataset


### resnet50
CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --variations --repeats 8 --name "b64 exp 8x"


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --repeats 24 --name "b64 baseline 24x"

### resnet101
CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet101 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --variations --repeats 8 --name "resnet101 b64 exp 8x"


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet101 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --repeats 24 --name "resnet101 b64 baseline 24x"


### VIT
CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=vit_base_patch16_224_miil_in21k --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --variations --repeats 8 --name "VIT b64 exp 8x"


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=vit_base_patch16_224_miil_in21k --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --repeats 24 --name "VIT b64 baseline 24x"

### Swin Transformer
CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=swin_base_patch4_window7_224 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --variations --repeats 8 --name "Swin b64 exp 8x"


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=swin_base_patch4_window7_224 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --repeats 24 --name "Swin b64 baseline 24x"


## N way example

CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1 --variations --repeats 200 --name "5 way b64 exp 200x" --classes 0 1 2 3 4


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/sun397 --dataset torch/sun397 -b=64 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=/home/judah/control_augmented_images_sun397_512 --diffaug-fewshot=1  --repeats 600 --name "5 way b64 baseline 600x" --classes 0 1 2 3 4






--diffaug-dir=./control_augmented_images_depth is the directory where the augmented images are stored. The images are loaded in from there.

--diffaug-fewshot=20 is the number of fewshot images to use. This will take the subset of the augmented images and use them for fewshot learning. (If --variations is set, the dataset will include the variations from the orignal images so 3x the number given here will be in the dataset)

--variations is a flag that tells the model to use the variations from the augmented dataset. If this flag is not set, the model will train only on the original images from the original dataset.

--repeats number of times to repeat the data in an epoch. useful for few shot, and to make baseline comparable in terms of compute

--classes

#### Cifar100
CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" 

#### Cifar10
CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar10 --dataset torch/cifar10 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" 

### Sun 397 Fewshot examples



### Diffusion entrypoint: see duffusion_augmentaion.py, called in line 658 in train.py

### Explaining Args
CUDA_VISIBLE_DEVICES specifies which GPUs are visible.

-b is batch size

--img-size is image size. For transfer learning, needs to be same as imagenet usually.

--epochs number of epochs

--color-jitter=0 --amp --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 these are used for data augmentation and preprocessing. Need not be touched.

--lr the learning rate

--sched lr schedule. cosine is a good choice

--model what model to use.

--pretrained use preloaded weights

--num-classes how many classes are in dataset

--opt the optimizer. SGD is good. Can try adam or adamw

--weight-decay weight decay

--dataset the dataset. if you set --download flag, it will download the dataset to download a dataset:

right after train.py, write "torch/<dataset_name>"

then write -d "torch/<dataset_name>"

To see all valid models we can use, run:
timm.list_models()
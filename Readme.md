# Diffusion Agumentation
## Judah Goldfeder, Gabe Guo, Gabriel Trigo, Patrick Puma, Hod Lipson

### Requirments
conda env create -f environment.yaml --prefix=/data/puma_envs/diffusion_augment

conda activate diffusion_augment


### Example script

CUDA_VISIBLE_DEVICES=2 python train.py torch/sun397 --dataset torch/sun397 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-no-variations" --diffaug-dir=./control_augmented_images_depth --diffaug-fewshot=20 --variations --

### Sun397 with augmentation

## First Create the dataset
CUDA_VISIBLE_DEVICES=0 python train.py torch/caltech256 --dataset torch/caltech256 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" --diffaug --diffaug-dir=./control_augmented_images_caltech256

--diffaug is a flag that tells the model to use diffusion augmentation. This will create the augmented images and store them in the directory specified by --diffaug-dir. Once the dataset is created, the images will be loaded in from this directory and the program will end.

--diffaug-dir=./control_augmented_images_depth is the directory where the augmented images are stored. This is the directory that is created when the dataset is created.

## Then train with the dataset
CUDA_VISIBLE_DEVICES=2 python train.py torch/sun397 --dataset torch/sun397 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=397 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "fewshot-variations" --diffaug-dir=./control_augmented_images_depth --diffaug-fewshot=20 --variations

--diffaug-dir=./control_augmented_images_depth is the directory where the augmented images are stored. The images are loaded in from there.

--diffaug-fewshot=20 is the number of fewshot images to use. This will take the subset of the augmented images and use them for fewshot learning. (If --variations is set, the dataset will include the variations from the orignal images so 3x the number given here will be in the dataset)

--variations is a flag that tells the model to use the variations from the augmented dataset. If this flag is not set, the model will train only on the original images from the original dataset.


#### Cifar100
CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" 

#### Cifar10
CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar10 --dataset torch/cifar10 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" 


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
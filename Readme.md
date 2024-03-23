# Diffusion Agumentation
## Judah Goldfeder, Gabe Guo, Gabriel Trigo, Patrick Puma, Hod Lipson

### Requirments
conda env create -f environment.yaml --prefix=/data/puma_envs/diffusion_augment

conda activate diffusion_augment


### Example script

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
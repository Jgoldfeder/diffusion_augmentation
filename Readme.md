# Diffusion Agumentation
## Judah Goldfeder, Gabe Guo, Gabriel Trigo, Patrick Puma, Hod Lipson

### Requirments
pip install torch
pip install timm
pip install wandb


### Example script
CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "test" 



### Diffusion entrypoint: see duffusion_augmentaion.py, called in line 658 in train.py
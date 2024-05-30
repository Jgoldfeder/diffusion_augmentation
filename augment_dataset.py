import argparse
import diffusion_augment
from dataset_factory import create_dataset


config_parser = parser = argparse.ArgumentParser(
    description="Training Config", 
    add_help=False
    )


# Dataset parameters
group = parser.add_argument_group('Dataset parameters')

group.add_argument(
    "--data-dir",
    metavar="NAME",
    default="",
    help="Root directory of the dataset"
)

group.add_argument(
    "--dataset", 
    metavar="NAME", 
    default="",
    help="dataset type + name ('<type>/<name>') \
        (default: ImageFolder or ImageTar if empty)"
    )

group.add_argument(
    "--train-split", 
    metavar="NAME", 
    default="train",
    help="dataset train split (default: train)"
    )

group.add_argument(
    "--dataset-download", 
    action="store_true", 
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that \
        support it."
    )

group.add_argument(
    "--diffaug-dir", 
    default="", 
    type=str,
    help="path to folder with diffusion augmentation files"
    )

group.add_argument(
    "--diffaug-fewshot", 
    default=2, 
    type=int,
    help="Number of fewshot images if you want to augment/use depending \
        on whether --diffaug flag is present (default is off)."
    )

group.add_argument(
    "--variations", 
    type=int, 
    default=0, 
    metavar="S",
    help="random seed (default: 42)"
    )

group.add_argument(
    "--images-per-class", 
    default=0, 
    type=int,
    help="Number of images per class for fewshot"
    )

group.add_argument(
    "--preprocessor", 
    default='Canny', 
    type=str, 
    help="Preprocessor to use for ControlNet either Canny or \
        Depth-Anything. Default is Canny"
    )

group.add_argument(
    "--diffusion-upscale", 
    action="store_true", 
    default=False,
    help="Diffusion upscale."
    )

group.add_argument(
    "--diffaug-resolutions", 
    default=512, 
    type=int,
    help="Resolutions for diffusion augmentation."
    )

group.add_argument(
    "--seed", 
    type=int, 
    default=42, 
    metavar="S",
    help="random seed (default: 42)"
    )

group.add_argument(
    "--no-controlnet", 
    action="store_true", 
    default=False,
    help="No controlnet."
    )

# args = _parse_args()[0].__dict__

args = config_parser.parse_known_args()[0]

print(args)
print(args.data_dir)
dataset_train, classes = create_dataset(
    args.dataset,
    root=args.data_dir,
    split=args.train_split,
    download=args.dataset_download,
    seed=args.seed,
)


dataset_train = diffusion_augment.augment(
    dataset_train, 
    preprocessor=args.preprocessor, 
    control_dir=args.diffaug_dir, 
    variations=args.diffaug_fewshot, 
    images_per_class=args.images_per_class, 
    res=args.diffaug_resolutions, 
    classes=classes,
    bad_aug=args.no_controlnet
    )

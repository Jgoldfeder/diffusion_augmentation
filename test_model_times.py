import time
from PIL import Image
from torchvision import transforms
from pathlib import Path
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager


transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])

def test_model_times():

    segment_aug_manager = SegmentAugmentationManager()
    color_aug_manager = ColorControlNetAugmentationManager()
    canny_aug_manager = CannyAugmentationManager()
    nerf_aug_manager = NerfAugmentationManager()
    depth_aug_manager = DepthAugmentationManager()

    aug_managers = [segment_aug_manager, color_aug_manager, canny_aug_manager, nerf_aug_manager, depth_aug_manager]

    image_path = Path("torch/caltech256/256_ObjectCategories/001.ak47/001_0001.jpg")
    image = Image.open(image_path)
    image = transform(image)
    
    for aug_manager in aug_managers:
        print(f"\nTesting {aug_manager}...")
        
        # Load model
        start_time = time.time()
        _ = aug_manager.generate_augmentations([image], [0])
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time} seconds")

if __name__ == "__main__":
    test_model_times()

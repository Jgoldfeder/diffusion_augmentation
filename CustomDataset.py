import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256, SUN397
from torch.utils.data import Dataset
from PIL import Image
import os
from augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from augmentation_models.NerfAugmentation import NerfAugmentationManager
from augmentation_models.DepthAugmentation import DepthAugmentationManager
from augmentation_models.SegAugmentation import SegmentationAugmentationManager
from augmentation_models.CannyAugmentation import CannyAugmentationManager

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, duplicate=1, use_diffusion_aug=False, args=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.args = args
        self.use_diffusion_aug = use_diffusion_aug
        
        if use_diffusion_aug:
            self.augmented_images = self._generate_diffusion_augmentations()
            self.num_augmentations = sum([
                self.args.use_canny,
                self.args.use_depth,
                self.args.use_seg,
                self.args.use_color,
                self.args.use_nerf
            ])
            self.duplicate = self.num_augmentations + 1
        else:
            self.duplicate = duplicate
    
    def _generate_diffusion_augmentations(self):
        temp_paths = []
        for idx, img in enumerate(self.images):
            if hasattr(img, 'filename') and img.filename:
                temp_path = os.path.abspath(img.filename)
            else:
                temp_path = os.path.abspath(f'temp_img_{idx}.png')
                img.save(temp_path)
            temp_paths.append(temp_path)
        
        if self.args.use_canny:
            canny_manager = CannyAugmentationManager()
            canny_aug = canny_manager.generate_augmentations(temp_paths)
        if self.args.use_depth:
            depth_manager = DepthAugmentationManager()
            depth_aug = depth_manager.generate_augmentations(temp_paths)
        if self.args.use_seg:
            seg_manager = SegmentationAugmentationManager()
            seg_aug = seg_manager.generate_augmentations(temp_paths)
        if self.args.use_color:
            color_manager = ColorControlNetAugmentationManager()
            color_aug = color_manager.generate_augmentations(temp_paths)
        if self.args.use_nerf:
            nerf_manager = NerfAugmentationManager()
            nerf_aug = nerf_manager.generate_augmentations(temp_paths)
                
        augmented_images = {}
        for path in temp_paths:
            img_augs = []
            if self.args.use_canny and path in canny_aug:
                img_augs.append(canny_aug[path])
            if self.args.use_depth and path in depth_aug:
                img_augs.append(depth_aug[path])
            if self.args.use_seg and path in seg_aug:
                img_augs.append(seg_aug[path])
            if self.args.use_color and path in color_aug:
                img_augs.append(color_aug[path])
            if self.args.use_nerf and path in nerf_aug:
                img_augs.append(nerf_aug[path])
            augmented_images[path] = img_augs
                
        return augmented_images
    
    def __len__(self):
        return len(self.images) * self.duplicate
    
    def __getitem__(self, idx):
        if self.use_diffusion_aug:
            true_idx = idx // self.duplicate
            aug_idx = idx % self.duplicate
            
            if aug_idx == 0:
                image = self.images[true_idx]
            else:
                if hasattr(self.images[true_idx], 'filename') and self.images[true_idx].filename:
                    img_path = os.path.abspath(self.images[true_idx].filename)
                else:
                    img_path = os.path.abspath(f'temp_img_{true_idx}.png')
                
                image = self.augmented_images[img_path][aug_idx - 1]
            
            if self.transform:
                image = self.transform(image)
            return image, self.labels[true_idx]
        else:
            true_idx = idx // self.duplicate
            image = self.images[true_idx]
            if self.transform:
                image = self.transform(image)
            return image, self.labels[true_idx]

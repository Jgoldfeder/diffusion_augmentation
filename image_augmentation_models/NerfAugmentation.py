import sys
import os

zero123_dir = os.path.join(os.getcwd(), 'zero123')
sys.path.append(zero123_dir)

from zero123.nerf import load_model_from_config, generate_angles
from zero123.ldm.util import create_carvekit_interface
from zero123.ldm.models.diffusion.ddim import DDIMSampler as Zero123DDIMSampler
from omegaconf import OmegaConf
from PIL import Image

class NerfAugmentationManager:
    def __init__(self, zero123_device='cuda:0'):
        self.zero123_device = zero123_device
        self.zero123_model = self._initialize_zero123_models()
        self.angles = [
            ("right", 0, 15, 0),
        ]

    def _initialize_zero123_models(self):
        """Initialize Zero123 models and return as dictionary"""
        zero123 = {}
        config_path = './model_configs/sd-objaverse-finetune-c_concat-256.yaml'
        config = OmegaConf.load(config_path)

        model_path = "./models/105000.ckpt"
        model = load_model_from_config(config, model_path, self.zero123_device)
        model = model.to(self.zero123_device)

        carvekit_interface = create_carvekit_interface()

        zero123['model'] = model
        zero123['carvekit_interface'] = carvekit_interface 
        
        return zero123

    def generate_augmentations(self, images, classes=None):
        augmentations = []

        for original_image in images:
            img = original_image.convert("RGB").resize((512, 512))
            
            preprocessed_image, augmented_images = generate_angles(
                input_image=img,
                angles=self.angles,
                model=self.zero123_model['model'],
                carvekit_interface=self.zero123_model['carvekit_interface'],
                device=self.zero123_device,
                precision='autocast',
                h=256,
                w=256,
                ddim_steps=50,
                scale=3.0,
                n_samples=1,
                ddim_eta=1.0
            )

            augmentations.append(augmented_images[0])
        return augmentations

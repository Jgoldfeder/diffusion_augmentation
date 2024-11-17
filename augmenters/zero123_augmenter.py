import os
import sys

zero123_dir = os.path.join("/home/shreyes", 'diffusion_augmentation', 'zero123')
sys.path.append(zero123_dir)

from zero123.nerf import load_model_from_config, generate_angles
from zero123.ldm.util import create_carvekit_interface
from zero123.ldm.models.diffusion.ddim import DDIMSampler as Zero123DDIMSampler
from omegaconf import OmegaConf


def initialize_zero123_models(zero123_device):
    """
    Initialize all models and detectors used in the pipeline
    Returns:
        control_net: dict, containing all ControlNet models and detectors
        llava: dict, containing the LLAVA model and processor
        zero123: dict, containing the Zero123 model and Carvekit interface
        color_control: dict, containing the Color Control model and SAM annotator

    """

    # Zero123 models
    # print('Loading Zero123 models...')
    zero123 = {}
    config_path = './model_configs/sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config_path)

    model_path = "./models/105000.ckpt"
    model = load_model_from_config(config, model_path, zero123_device)
    model = model.to(zero123_device)

    # print('Creating Carvekit interface...')
    carvekit_interface = create_carvekit_interface()

    zero123['model'] = model
    zero123['carvekit_interface'] = carvekit_interface 
    
    return zero123
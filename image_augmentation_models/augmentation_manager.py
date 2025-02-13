from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

class AugmentationManager:
	def __init__(self):
		self.segment_manager = SegmentAugmentationManager()
		self.color_manager = ColorControlNetAugmentationManager()
		self.canny_manager = CannyAugmentationManager()
		self.nerf_manager = NerfAugmentationManager()
		self.depth_manager = DepthAugmentationManager()
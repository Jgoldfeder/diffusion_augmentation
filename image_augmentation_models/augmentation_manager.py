import random
from enum import Enum

from image_augmentation_models.ControlNetAugmentation import ControlNetAugmentationManager
from image_augmentation_models.SegmentAugmentation import SegmentAugmentationManager
from image_augmentation_models.ColorControlNetAugmentation import ColorControlNetAugmentationManager
from image_augmentation_models.CannyAugmentation import CannyAugmentationManager
from image_augmentation_models.NerfAugmentation import NerfAugmentationManager
from image_augmentation_models.DepthAugmentation import DepthAugmentationManager

class AugmentationType(Enum):
	CANNY = 0
	DEPTH = 1
	SEGMENT = 2
	COLOR = 3
	NERF = 4
	CLASSICAL = 5
	NONE = 6

	def get_random_augmentation():
		return random.choice(list(AugmentationType))

# NOTE may want to add classical and none as class versions which match the pattern
# to make things more elegant
class AugmentationManager:
	_managers: dict[AugmentationType, ControlNetAugmentationManager | NerfAugmentationManager | ColorControlNetAugmentationManager] = {}
	_initialized = False

	@classmethod
	def initialize(cls):
		if not cls._initialized:
			cls._managers = {
				AugmentationType.SEGMENT: SegmentAugmentationManager(),
				AugmentationType.COLOR: ColorControlNetAugmentationManager(),
				AugmentationType.CANNY: CannyAugmentationManager(),
				AugmentationType.NERF: NerfAugmentationManager(),
				AugmentationType.DEPTH: DepthAugmentationManager()
			}
			cls._initialized = True

	@classmethod
	def get_manager(cls, augmentation_type: AugmentationType):
		cls.initialize()
		return cls._managers.get(augmentation_type)

if __name__ == '__main__':
	from PIL import Image

	AugmentationManager.initialize()
	print(AugmentationManager)
	AugmentationManager.initialize()

	img = Image.open('orig_images/0.png')

	seg_am = AugmentationManager.get_manager(AugmentationType.SEGMENT)

	breakpoint()
if __name__ == "__main__":
	import dataset_manager
	import augmentation_tree
	
	aug_tree = augmentation_tree.BinaryAugmentationNode()
	aug_tree.augmentation_type = augmentation_tree.AugmentationType.NONE
	aug_tree.left_probability = 0.5
	aug_tree.left = augmentation_tree.BinaryAugmentationNode()
	aug_tree.right = augmentation_tree.BinaryAugmentationNode()
	aug_tree.left.augmentation_type = augmentation_tree.AugmentationType.CLASSICAL
	aug_tree.right.augmentation_type = augmentation_tree.AugmentationType.NONE
	aug_dataset = augmentation_tree.TreeAugmentedDataset(dataset_manager.get_dataset_path('caltech256', 5, 2, 42, train=True), aug_tree, 1)

	train_dataset, val_dataset = dataset_manager.split_train_val(aug_dataset)

	print(len(train_dataset))
	print(len(val_dataset))
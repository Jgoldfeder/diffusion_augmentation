import torchvision.transforms as transforms

from CustomDataset import FewShotDataset, ClassicalDataset


if __name__ == '__main__':
	train_dataset = FewShotDataset('few_shot_datasets/caltech256/2_shot/seed_41', dataset_type='train')

	print(len(train_dataset))
	print(train_dataset[0])

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	classical_datset = ClassicalDataset(train_dataset, transform, duplicate_factor=5)

	print(len(classical_datset))
	print(classical_dataset[0])
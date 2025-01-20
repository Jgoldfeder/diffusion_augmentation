from torchvision.datasets import Caltech256

if __name__ == '__main__':
	# read in caltech 256 dataset

	dataset_name = 'caltech256'
	seed = 42
	num_ways = 5 # will always be 5
	num_shots = 2

	dataset = Caltech256(root='./torch')

	class_to_label = dict()
	label_to_class = dict()
	for category in dataset.categories:
		parts = category.split('.')
		label = int(parts[0]) - 1
		class_name = parts[1]
		class_to_label[class_name] = label
		label_to_class[label] = class_name
		labels = [label for _, label in dataset]
	
	breakpoint()
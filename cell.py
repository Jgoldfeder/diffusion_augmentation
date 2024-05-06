from torchvision.datasets import Caltech256, ImageFolder

# og_dataset = Caltech256(root = "./torch" ,download=True,transform=None)
# og_dataset = ImageFolder(root="/home/augmented_data/caltech-pat/",transform=None)
# og_dataset[0][0].save("./pat_example_imagefolder.png")
Caltech256(root = "/home/augmented_data/pat" ,download=True,transform=None)
import torchvision
import numpy as np
from PIL import Image   

judah = torchvision.datasets.ImageFolder(root = "/home/augmented_data/judah/caltech256/256_ObjectCategories/" ,transform=None)
pat = torchvision.datasets.ImageFolder(root = "/home/augmented_data/pat/caltech256/256_ObjectCategories/" ,transform=None)
print(np.array_equal(np.array(judah[0][0]), np.array(pat[0][0])))


print(np.array_equal(np.array(Image.open("/home/augmented_data/pat_example.png")), np.array(pat[0][0])))


print(np.array_equal(np.array(Image.open("/home/augmented_data/judah_example.png")), np.array(pat[0][0])))

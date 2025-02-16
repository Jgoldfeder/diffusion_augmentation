When we create folds, we are doing it off of the dataset which is created after doing the augmentations from the tree not from the dataset which is loaded from the folder

why do we upsize the images before doing the image augmentations

Tasks to complete:
* Fix the genome to tree conversion -- should match the original depth 4 tree strucutre with correct probabilities
* Ensure the number of fitness function calls is intended (Pygad is weird with repeated fitness function calls) 
* Memory issue with loading models -- runs get killed (this should also fix the k fold cross validation issue)
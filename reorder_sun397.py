import os
import shutil

def concatenate_folders(folder_path):
    folder_names = []
    while True:
        folder_path, folder_name = os.path.split(folder_path)
        if folder_name == "":
            break
        if len(folder_name) > 1:
            folder_names.append(folder_name)
    return '_'.join(folder_names[::-1])

def copy_and_restructure_dataset(root_dir, output_dir):
    count = 0
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        print(class_path)
        if os.path.isdir(class_path):
            # If there are subdirectories within the class folder
            subfolders = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
            print(subfolders)
            if len(subfolders) > 0:
                ogsubfolders = subfolders.copy()
                # check each subfolder if they have more classes inside
                removed = False
                for sub in ogsubfolders:
                    for dir in os.listdir(os.path.join(class_path, sub)):
                        if os.path.isdir(os.path.join(class_path, sub, dir)):
                            print(sub)
                            subfolders.append(os.path.join(sub, dir))
                            if not removed:
                                subfolders.remove(sub)
                                removed = True
                    removed = False

                
                
                # Copy all files from subfolders to the new class folder
                for subfolder in subfolders:
                    subfolder_path = os.path.join(class_path, subfolder)
                    # Concatenate the names of all parent folders
                    new_class_name = "_".join(subfolder.split("/"))
                    
                    new_class_path = os.path.join(output_dir, new_class_name)
                    print(new_class_path)
                    # Create the new class folder if it doesn't exist
                    if not os.path.exists(new_class_path):
                        os.makedirs(new_class_path)
                    for file in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file)
                        # Check if it's a file before copying
                        if os.path.isfile(file_path):
                            shutil.copy(file_path, new_class_path)
                    count+=1
    print(count)
                
# Example usage:

root_directory = "./torch/sun397/SUN397"
output_directory = "./torch/restructured_sun397"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

copy_and_restructure_dataset(root_directory, output_directory)

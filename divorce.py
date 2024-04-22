import os
import zipfile
import tempfile
from sklearn.model_selection import train_test_split
import shutil

def extract_frames_in_place(root_dir):
    for season in ['fall']: #, 'spring', 'winter']:
        for data_type in ['color_left', 'depth']:
            base_dir = os.path.join(root_dir, 'PLE_training', season, data_type)
            for trajectory in os.listdir(base_dir):
                zip_path = os.path.join(base_dir, trajectory, 'frames.zip')
                if os.path.isfile(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(base_dir, trajectory))
                        print(f"Extracted {zip_path}")
                extracted_folder_path = os.path.join(base_dir, trajectory, 'frames.zip_extracted')
                
                # Check if the extracted folder exists and remove it
                if os.path.isdir(extracted_folder_path):
                    shutil.rmtree(extracted_folder_path)
                    print(f"Removed existing directory {extracted_folder_path}")

def collect_data_pairs(root_dir):
    pairs = []
    for season in ['fall']: #, 'spring', 'winter']:
        color_dir = os.path.join(root_dir, 'PLE_training', season, 'color_left')
        depth_dir = os.path.join(root_dir, 'PLE_training', season, 'depth')
        
        for trajectory in os.listdir(color_dir):
            color_trajectory_dir = os.path.join(color_dir, trajectory)
            depth_trajectory_dir = os.path.join(depth_dir, trajectory)
            
            # Assuming the same naming convention for color and depth images,
            # except for the file extension.
            for image_file in os.listdir(color_trajectory_dir):
                if image_file.endswith('.JPEG'):  # Assuming all images are .jpg
                    color_path = os.path.join(color_trajectory_dir, image_file)
                    depth_file = image_file.replace('.JPEG', '.PNG')  # Assuming depth files are .png
                    depth_path = os.path.join(depth_trajectory_dir, depth_file)
                    
                    # Save the relative paths or adjust as necessary.
                    pairs.append((color_path, depth_path))
                    
    return pairs

def split_data(pairs, test_size=0.1):
    train, test = train_test_split(pairs, test_size=test_size, random_state=42)
    return train, test

def write_to_file(pairs, file_path, focal_length=512.0): # Assuming a focal length of 512.0
    with open(file_path, 'w') as f:
        for color_path, depth_path in pairs:
            # Extracting file names without the zip extension
            color_file_name = color_path
            depth_file_name = depth_path
            
            line = f"{color_file_name} {depth_file_name} {focal_length}\n"
            f.write(line)

root_dir = '/home/judah/diffusion_augmentation/MidAir'

# Step 1: Extract frames.zip files in place
# Uncomment this line to extract the files
# extract_frames_in_place(root_dir)

# Step 2: Collect data pairs
pairs = collect_data_pairs(root_dir)
print(f"Collected {len(pairs)} data pairs.")

# Step 3: Split data into training and testing sets
train_pairs, test_pairs = split_data(pairs)

# Step 4: Write the data pairs to text files
write_to_file(train_pairs, 'train_data.txt')
write_to_file(test_pairs, 'test_data.txt')

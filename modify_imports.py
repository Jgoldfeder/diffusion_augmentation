import os
import re

def process_import_line(line):
    # Match import statements starting with 'from annotator' or 'import annotator'
    annotator_import_pattern = r'^(from|import)\s+annotator\.(.*)'
    match = re.match(annotator_import_pattern, line.strip())
    
    if match:
        import_statement = match.group(0)  # Full import statement
        rest_of_line = match.group(2)  # Part after 'annotator.'
        
        # Format the try-except block
        try_except_block = (
            f"try:\n"
            f"    {import_statement}\n"
            f"except ModuleNotFoundError:\n"
            f"    {match.group(1)} controlnet.annotator.{rest_of_line}\n"
        )
        return try_except_block
    return line

def process_file(file_path):
    # Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line in the file
    new_lines = []
    for line in lines:
        new_lines.append(process_import_line(line))

    # Write the modified file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

def process_folder(folder_path):
    # Recursively walk through the folder and process all .py files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f'Processing {file_path}')
                process_file(file_path)

if __name__ == "__main__":
    folder_path = "/home/pat/diffusion_augmentation/controlnet/annotator/uniformer/mmseg/models/losses/"
    process_folder(folder_path)
    print("Processing completed.")

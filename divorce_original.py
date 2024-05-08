def extract_original_images(input_file, output_file):
    """
    Extracts lines containing the "original" image variation and writes them to an output file.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if "_original.png" in line:
                file.write(line)

def main():
    training_input = 'train_aug.txt'
    testing_input = 'test_aug.txt'
    training_output = 'train_data.txt'
    testing_output = 'test_data.txt'

    extract_original_images(training_input, training_output)
    extract_original_images(testing_input, testing_output)
    print("Original images extracted for both training and testing datasets.")

if __name__ == "__main__":
    main()
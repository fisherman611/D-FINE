import os
# Define the directory path
output_dir = "Fisheye8K/Fisheye8K_all_including_train&test"
# Define the file name you want to create
file_name = "annotation"
# Construct the full path to the new file
output_filepath = os.path.join(output_dir, file_name)

try:
    # Create the directory if it does not exist (including any necessary parent directories)
    os.makedirs(output_dir, exist_ok=True)

    # Create the empty file
    with open(output_filepath, 'w') as f:
        # You can write some content here if you want, otherwise it will be an empty file
        # f.write("This is my new annotation file.\n")
        pass # Creates an empty file

    print(f"Successfully created the file: {output_filepath}")

except Exception as e:
    print(f"An error occurred: {e}")
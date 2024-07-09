import os

# Function to check if a file exists and delete it
def check_and_delete_file(file_path):
    # Check if the given path is just a file name
    if not os.path.isabs(file_path):
        # Prepend the current working directory to the file name
        file_path = os.path.join(os.getcwd(), file_path)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing file: {file_path}")
import csv
import os

# Define the directory containing your Python files
directory = os.getcwd()

# Get a list of Python files in the directory
python_files = [file for file in os.listdir(directory) if file.endswith('optimization.py')]  # change for any file to csv in directory

# Prepare a list to store the file data
file_data = []

# Iterate over each Python file
for file_name in python_files:
    file_path = os.path.join(directory, file_name)
    
    # Read the code content from the file
    with open(file_path, 'r') as file:
        code_content = file.read()
    
    # Create a dictionary with the file name and code content
    file_dict = {'file_name': file_name, 'code_content': code_content}
    
    # Append the dictionary to the list
    file_data.append(file_dict)

# Define the CSV file path in the current directory
csv_file = os.path.join(os.getcwd(), 'output.csv')

# Write the file data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['file_name', 'code_content'])
    writer.writeheader()
    writer.writerows(file_data)
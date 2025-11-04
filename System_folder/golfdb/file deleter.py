#use not in for drive list, use row[1]
#use in for SlowSwingList, use row[2]
#use not in for down-the-line, use row[2]


import csv
import os

# File names to check against
filename = "data/DownTheLineList.csv"

# List to store potential names
potential_names = []

# Read the CSV file
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        potential_names.append(row[2])

# Directory to check for files
dir_path = "data/videos_160"

# Loop through all the files in the directory
for file_name in os.listdir(dir_path):
    # Check if the file name is in the potential names list
    if file_name not in potential_names:
        # If the file name is in the list, delete the file
        file_path = os.path.join(dir_path, file_name)
        os.remove(file_path)
        print(f"Deleted file: {file_name}")

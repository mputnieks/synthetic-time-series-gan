"""Clean and standartize the DIALECT data by removing irrelevant files and folders."""
import os
import shutil

# set the paths
path = "data-all"
path_cgm = "data-dialect"

# for each subfolder in the main folder...
dirs = os.listdir(path)
count = 0
for dir in dirs:
    subfolder = os.path.join(path, dir) # get the path to subfolder
    print("Processing: ", subfolder)
    
    for file in os.listdir(subfolder):
        # if any file name contains the case-insensitive letter combination "gluc"
        if "gl" in file.lower():
            
            # create a new subfolder in the "data with cgm" folder
            os.mkdir(os.path.join(path_cgm, dir))
            count += 1
            
            # copy relevant files from the subfolder to the new subfolder
            for file in os.listdir(subfolder):
                if (file.endswith(".xlsx") or file.endswith(".csv") or file.endswith(".txt")) and not file.startswith("."):
                    # check if file contains "gluc" or "step" in the name
                    if "gl" in file.lower() or "step" in file.lower():
                        shutil.copy(os.path.join(subfolder, file), os.path.join(path_cgm, dir))
            
            break # break the loop to avoid copying the same files multiple times

print("Done!")
print("Folders copied: ", count)
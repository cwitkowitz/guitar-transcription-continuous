# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import os

# Define the path to the base directory
base_dir = 'path/to/BaseDir'

# Choose the to/from patterns
str_from = '#'
str_to = 'sharp'

# Walk through the base directory
for root, dirs, files in os.walk(base_dir):
    # Loop through all files in the current directory
    for file_name in files:
        # Process files with the problematic string pattern
        if str_from in file_name:
            # Split the file name at the problematic pattern(s)
            splits = file_name.split(str_from)
            # Start building the new name with the first valid piece
            new_name = splits[0]

            for i in range(1, len(splits)):
                # Iteratively add all other valid pieces with the new pattern in between
                new_name += (str_to + splits[i])

            # Construct the old and new file path
            old_path = os.path.join(root, file_name)
            new_path = os.path.join(root, new_name)
            # Update the path with the new file name
            os.rename(old_path, new_path)

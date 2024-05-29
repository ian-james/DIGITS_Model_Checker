#!/bin/bash

# Create a function to parse and sort the file
sort_file() {
    # Get the filename
    filename="$1"
    
    # Extract the md, mt, and mp values using grep and sed
    md_value=$(echo "$filename" | grep -oP '(?<=_md_)[^_]+')
    mt_value=$(echo "$filename" | grep -oP '(?<=_mt_)[^_]+')
    mp_value=$(echo "$filename" | grep -oP '(?<=_mp_)[^_]+')
    
    # Create the target directory path
    target_dir="sorted/md_${md_value}_mt_${mt_value}_mp_${mp_value}"
    
    # Create the directory if it does not exist
    mkdir -p "$target_dir"
    
    # Move the file to the target directory
    mv "$filename" "$target_dir/"
    
    echo "Moved $filename to $target_dir/"
}

# Check if at least one file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <filenames>"
    exit 1
fi

# Iterate over all provided files and sort them
for file in "$@"; do
    if [ -f "$file" ]; then
        sort_file "$file"
    else
        echo "File $file does not exist."
    fi
done


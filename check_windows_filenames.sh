#!/usr/bin/env bash

# Initialize counters for valid and invalid files
valid_count=0
invalid_count=0

# Function to validate a file name against Windows standards
validate_file_name() {
    local file_path="$1"
    local file_name=$(basename "$file_path")

    # Check if the file is ignored by .gitignore
    if git check-ignore -q "$file_path"; then
        return  # Skip files ignored by .gitignore
    fi

    # Check for length (Windows path length limit is 255 characters)
    if [ ${#file_path} -gt 255 ]; then
        echo "INVALID (length): $file_path"
        ((invalid_count++))
        return
    fi

    # Check for invalid characters using a case statement
    case "$file_name" in
        *[\:\\/\*\?\"\<\>\|]*) 
            echo "INVALID (characters): $file_path"
            ((invalid_count++))
            return
            ;;
    esac

    # If valid
    echo "VALID: $file_path"
    ((valid_count++))
}

# Recursive function to check all files in a directory
check_files_recursively() {
    local dir="$1"

    # Iterate through all items in the directory
    for item in "$dir"/*; do
        if [ -f "$item" ]; then
            # Validate file name
            validate_file_name "$item"
        elif [ -d "$item" ]; then
            # Recursively check subdirectory
            check_files_recursively "$item"
        fi
    done
}

# Determine the target directory
if [ $# -eq 0 ]; then
    # Use the directory where this script is located
    target_directory=$(dirname "$0")
else
    # Use the directory passed as argument
    target_directory="$1"
fi

# Verify if the directory exists
if [ ! -d "$target_directory" ]; then
    echo "Error: Directory '$target_directory' does not exist."
    exit 1
fi

# Change to the target directory to ensure .gitignore is respected
cd "$target_directory" || exit

# Start checking files
echo "Checking files in directory: $target_directory"
check_files_recursively "$target_directory"

# Report the results
echo
echo "Summary:"
echo "Valid files: $valid_count"
echo "Invalid files: $invalid_count"


#!/usr/bin/env python3
# package.py

import os
import tarfile
import datetime

# --- Configuration ---
# The directory containing the source files.
SOURCE_DIR = '.' 
# The directory where the final package will be saved.
BACKUP_DIR = 'backups'
# The name of the output tarball.
OUTPUT_FILENAME = f"aicli_package_{datetime.datetime.now().strftime('%Y%m%d')}.tar.gz"
# The full path for the output file.
OUTPUT_PATH = os.path.join(BACKUP_DIR, OUTPUT_FILENAME)
# The name of the root folder inside the tarball.
ARC_DIR_NAME = 'aicli'

# Directories and files to exclude from the package.
EXCLUDE_DIRS = {'Test', 'prompts', 'logs', 'images', '__pycache__', '.git', BACKUP_DIR}
EXCLUDE_FILES = {'package.py', '.env'} # The packaging script itself

# --- Script Logic ---

def exclude_filter(tarinfo):
    """
    Filter function to be used with tarfile.add().
    Returns None for items that should be excluded.
    """
    # Exclude the file if its name is in the exclusion list
    if os.path.basename(tarinfo.name) in EXCLUDE_FILES:
        print(f"  Excluding file: {tarinfo.name}")
        return None
        
    # Check if any part of the path is an excluded directory
    path_parts = tarinfo.name.split(os.sep)
    if any(part in EXCLUDE_DIRS for part in path_parts):
        print(f"  Excluding path: {tarinfo.name}")
        return None
        
    print(f"  Adding: {tarinfo.name}")
    return tarinfo

def create_package():
    """Finds all project files and creates a compressed tarball."""
    print(f"Creating package: {OUTPUT_PATH}...")
    
    # Ensure the backup directory exists
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
    except OSError as e:
        print(f"\nError: Could not create backup directory '{BACKUP_DIR}': {e}")
        return

    try:
        with tarfile.open(OUTPUT_PATH, "w:gz") as tar:
            # Add the source directory to the archive, using the filter
            tar.add(SOURCE_DIR, arcname=ARC_DIR_NAME, filter=exclude_filter)
        
        print("\nPackage created successfully!")
        print(f" -> {OUTPUT_PATH}")
    except Exception as e:
        print(f"\nError: Failed to create package: {e}")

if __name__ == "__main__":
    create_package()

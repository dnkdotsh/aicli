#!/usr/bin/env python3
# sync.py

import os
import sys
import subprocess
import argparse
import datetime
from pathlib import Path

# --- Configuration ---
MAIN_DIR = Path.home() / 'aicli'
TEST_DIR = Path.home() / 'aicli-test'

# --- Utility Functions ---

def run_command(command, cwd=None, stdin_input=None):
    """Executes a shell command with optional stdin and returns True on success."""
    try:
        if command[0] == 'rsync':
            process = subprocess.run(command, check=True, cwd=cwd)
        else:
            process = subprocess.run(
                command, check=True, cwd=cwd,
                capture_output=True, text=True
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}\n{e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it in your PATH?", file=sys.stderr)
        return False

def check_git_status(directory):
    """Checks if a git repository has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, check=True, cwd=directory
        )
        return result.stdout.strip() != ''
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# --- Sync Logic ---

def sync_to_test():
    """Syncs essential source files from the main project to the test directory."""
    print(f"Preparing to sync essential source from '{MAIN_DIR}' to '{TEST_DIR}'...")

    if check_git_status(TEST_DIR):
        print("\nWarning: The test directory has uncommitted Git changes.")
        choice = input("These changes may be overwritten. Continue? (y/N): ").lower()
        if choice != 'y':
            print("Sync cancelled.")
            sys.exit(0)

    print("\nSyncing to test directory...")
    print("----------------------------------------------------")
    
    include_list = [
        'src/**',
        'tests/**',
        'pyproject.toml',
        'requirements.txt',
        '.env.example'
    ]

    rsync_command = [
        'rsync',
        '-avm', '--progress', '--delete',
        # Add specific excludes before general includes
        '--exclude=*.egg-info',
        '--exclude=__pycache__',
        # This rule allows rsync to traverse into directories
        '--include=*/'
    ]
    
    for path in include_list:
        rsync_command.append(f'--include={path}')
    
    rsync_command.append('--exclude=*')
    
    rsync_command.extend([str(MAIN_DIR) + '/', str(TEST_DIR) + '/'])
    
    run_command(rsync_command)
    
    print("----------------------------------------------------")
    print("Sync to test directory complete.")

def sync_from_test(dry_run=False, push_to_git=False):
    """Syncs from the test directory back to the main project directory."""
    print(f"Preparing to sync from '{TEST_DIR}' to '{MAIN_DIR}'...")

    if not dry_run:
        print("\nWARNING: This will overwrite files in your main project directory.")
        choice = input("Are you sure you want to proceed? (y/N): ").lower()
        if choice != 'y':
            print("Sync cancelled.")
            sys.exit(0)
    
    rsync_flags = ['-avu', '--progress']
    if dry_run:
        rsync_flags.append('--dry-run')
        print(">>> Performing a DRY RUN. No files will be changed.")
    
    print("\nSyncing from test directory...")
    print("----------------------------------------------------")
    rsync_command = [
        'rsync', *rsync_flags,
        '--exclude=*.egg-info',
        '--exclude=__pycache__/', '--exclude=logs/', '--exclude=backups/',
        '--exclude=.git/', '--exclude=.venv/',
        str(TEST_DIR) + '/', str(MAIN_DIR) + '/'
    ]
    
    if not run_command(rsync_command):
        sys.exit(1)
        
    print("----------------------------------------------------")
    if dry_run:
        print(">>> Dry run complete.")
        return

    print("Sync from test directory complete.")

    if push_to_git:
        print("\n--- Starting Git Sync ---")
        if not check_git_status(MAIN_DIR):
            print("Git working directory is clean. Nothing to sync.")
            return

        print("Staging changes...")
        if not run_command(['git', 'add', '.'], cwd=MAIN_DIR): return

        commit_message = f"Auto-sync from test environment on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"Committing with message: '{commit_message}'")
        if not run_command(['git', 'commit', '-m', commit_message], cwd=MAIN_DIR): return

        print("Pushing to remote repository...")
        if not run_command(['git', 'push'], cwd=MAIN_DIR): return
        
        print("Git sync complete.")
    else:
        print("Skipping git push as requested.")

# --- Main Execution ---

def main():
    """Parses arguments and orchestrates the sync operations."""
    parser = argparse.ArgumentParser(
        description="A script to synchronize changes between a main and test directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='direction', required=True, help='The direction of the sync.')

    parser_to_test = subparsers.add_parser('to-test', help='Sync essential source FROM main TO the test directory.')
    parser_to_test.set_defaults(func=sync_to_test)

    parser_from_test = subparsers.add_parser('from-test', help='Sync ALL changes FROM test back TO the main project directory.')
    parser_from_test.add_argument('-n', '--dry-run', action='store_true', help='Perform a dry run without making changes.')
    parser_from_test.add_argument('--push', action='store_true', help='Commit and push changes to Git after syncing.')
    parser_from_test.set_defaults(func=lambda args: sync_from_test(args.dry_run, args.push))

    args = parser.parse_args()
    
    if not MAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        print(f"Error: Ensure both '{MAIN_DIR}' and '{TEST_DIR}' exist.", file=sys.stderr)
        sys.exit(1)
        
    if 'func' in args:
        if args.direction == 'from-test':
            args.func(args)
        else:
            args.func()

if __name__ == "__main__":
    main()

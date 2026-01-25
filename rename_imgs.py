import os
from pathlib import Path

def renumber_images(folder_path: Path, ask_confirmation = True):
    """
    Renumbers PNG files in a folder to close gaps in numbering sequence.
    Starting from 1.png, 2.png, 3.png, etc.
    
    Args:
        folder_path: Path to the folder containing PNG files
    """
    
    # Get all PNG files with numeric names
    png_files = []
    for file in folder_path.glob("*.png"):
        try:
            # Extract the number from filename
            num = int(file.stem)
            png_files.append((num, file))
        except ValueError:
            # Skip files that don't have numeric names
            continue
    
    # Sort by the numeric value
    png_files.sort(key=lambda x: x[0])
    
    if not png_files:
        print("No numbered PNG files found in the folder.")
        return
    
    print(f"Found {len(png_files)} numbered PNG files")
    print("\nRenaming plan:")
    
    # Create a renaming plan
    rename_plan = []
    for new_num, (old_num, file) in enumerate(png_files, start=1):
        if old_num != new_num:
            rename_plan.append((file, new_num))
            print(f"  {file.name} -> {new_num}.png")
    
    if not rename_plan:
        print("No renaming needed - files are already numbered sequentially!")
        return
    
    # Ask for confirmation
    if ask_confirmation:
        response = input("\nProceed with renaming? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Renaming cancelled.")
            return
    
    # Perform renaming in two passes to avoid conflicts
    # First pass: rename to temporary names
    temp_files = []
    for file, new_num in rename_plan:
        temp_name = folder_path / f"temp_{new_num}.png"
        file.rename(temp_name)
        temp_files.append((temp_name, new_num))
    
    # Second pass: rename to final names
    for temp_file, new_num in temp_files:
        final_name = folder_path / f"{new_num}.png"
        temp_file.rename(final_name)
    
    print(f"\nSuccessfully renumbered {len(rename_plan)} files!")

if __name__ == "__main__":
    # Get folder path from user
    folder_path = input("Enter the folder path containing PNG files: ").strip()
    
    # Remove quotes if user pastes path with quotes
    folder_path = folder_path.strip('"').strip("'")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
    elif not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a folder.")
    else:
        renumber_images(folder_path)

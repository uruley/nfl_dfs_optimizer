import os

def print_directory_structure(root_dir, indent=""):
    """
    Recursively print the directory structure starting from root_dir.
    """
    try:
        # Get all items in the directory
        items = os.listdir(root_dir)
        items.sort()  # Sort for consistent output
        
        for item in items:
            item_path = os.path.join(root_dir, item)
            # Print directories with a trailing slash
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")
                # Recurse into subdirectory
                print_directory_structure(item_path, indent + "  ")
            else:
                # Print files
                print(f"{indent}📄 {item}")
    except PermissionError:
        print(f"{indent}[Permission Denied: {root_dir}]")
    except Exception as e:
        print(f"{indent}[Error accessing {root_dir}: {e}]")

if __name__ == "__main__":
    # Set the root directory to the current working directory
    # Change this to your DFS optimizer project folder path if needed
    root_directory = os.getcwd()  # Or specify like: "C:/path/to/your/project"
    print(f"Directory structure for: {root_directory}")
    print("-" * 50)
    print_directory_structure(root_directory)
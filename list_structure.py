import os

output_file = "project_structure.txt"
max_files_per_dir = 5

def dfs(path, prefix=""):
    entries = sorted(os.listdir(path))
    files = [f for f in entries if os.path.isfile(os.path.join(path, f))]
    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    # Limit number of files
    for i, f in enumerate(files[:max_files_per_dir]):
        with open(output_file, "a") as out:
            out.write(f"{prefix}├── {f}\n")

    # Recurse into subdirectories
    for i, d in enumerate(dirs):
        with open(output_file, "a") as out:
            out.write(f"{prefix}├── {d}/\n")
        dfs(os.path.join(path, d), prefix + "│   ")

# Clear previous output
open(output_file, "w").close()

# Start from current dir
dfs(".")

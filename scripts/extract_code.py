import os

def extract_python_files(repo_paths, extensions=(".py",)):
    code_blocks = []
    for repo_path in repo_paths:
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(extensions):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            code_blocks.append(f.read())
                    except Exception:
                        continue
    return code_blocks

if __name__ == "__main__":
    paths = ["repos/TheAlgorithms", "repos/cpython/Lib"]
    all_code = extract_python_files(paths)
    print(f"Extracted {len(all_code)} code files.")

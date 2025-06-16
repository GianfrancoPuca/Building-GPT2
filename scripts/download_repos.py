import os
import subprocess

repos = {
    "TheAlgorithms": "https://github.com/TheAlgorithms/Python.git",
    "cpython": "https://github.com/python/cpython.git"
}

def download_repos(base_path="repos"):
    os.makedirs(base_path, exist_ok=True)
    for name, url in repos.items():
        repo_path = os.path.join(base_path, name)
        if not os.path.exists(repo_path):
            print(f"Cloning {name}...")
            subprocess.run(["git", "clone", url, repo_path])
        else:
            print(f"{name} already cloned.")

if __name__ == "__main__":
    download_repos()

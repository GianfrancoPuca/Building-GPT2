from extract_code import extract_python_files
from clean_code import clean_code
import os

def build_dataset(output_path="data/input.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paths = ["repos/TheAlgorithms", "repos/cpython/Lib"]
    all_code = extract_python_files(paths)

    cleaned = [clean_code(code) for code in all_code if len(code.strip()) > 50]
    full_text = "\n\n".join(cleaned)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Dataset written to {output_path}. Length: {len(full_text)} characters.")

if __name__ == "__main__":
    build_dataset()

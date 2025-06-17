# 🧠 Mini-GPT for Python Code — From GitHub to Transformer

This project implements a **miniature GPT-style language model** trained on **Python code extracted from real GitHub repositories**. Inspired by Andrej Karpathy's nanoGPT series, the project goes beyond toy datasets like Shakespeare and focuses instead on generating structured, executable Python code.

---

## 📂 Project Structure

.
├── data/
│ └── input.txt # The full training corpus, extracted from GitHub repositories
├── scripts/
│ ├── download_repos.py # Clones GitHub repositories for data collection
│ ├── extract_code.py # Extracts .py files and concatenates them into a single text corpus
│ ├── clean_code.py # Optionally cleans and normalizes the code (whitespace, formatting, etc.)
│ └── build_dataset.py # Orchestrates the entire pipeline and generates input.txt
├── notebooks/
│ └── attention_playground.ipynb # Interactive notebook to visualize and understand attention
├── train.py # Full training script: from input.txt to trained model and generation
└── README.md

---

## 📥 Dataset Creation Pipeline

We build a training dataset (`input.txt`) from Python files in open-source repositories.

### Repositories Used
- [`TheAlgorithms/Python`](https://github.com/TheAlgorithms/Python)
- [`python/cpython`](https://github.com/python/cpython)

### Steps

1. **Clone Repositories**
   ```bash
   python scripts/download_repos.py

Extract Python Files

python scripts/extract_code.py

(Optional) Clean Code

python scripts/clean_code.py
Build Dataset

python scripts/build_dataset.py

This generates data/input.txt, a character-level corpus of Python source code.

📓 Notebooks
attention_playground.ipynb
An interactive notebook that walks through the core mechanics of attention:

Causal aggregation via manual loops and matrix operations

Uniform averaging, softmax attention, and full learnable self-attention

Scaled dot-product attention using Q, K, V projections

Visualization and shape inspection to fully understand self-attention dynamics

Useful for educational purposes or prototyping new attention mechanisms.

🚀 Training the Model
train.py
A full training script that defines and trains a mini-GPT Transformer model on the input.txt corpus.

Features:
Character-level tokenization from scratch

Positional and token embeddings

Transformer blocks with multi-head self-attention

Autoregressive training with causal masking

Configurable hyperparameters (batch size, layers, etc.)

Text generation from a trained model

Run:
python train.py

This will:

Load data/input.txt

Train the model on Python code for a configurable number of steps

Print training and validation loss at intervals

Autoregressively generate a Python code snippet at the end

🧠 Goals and Next Steps
✅ Build dataset from real-world code

✅ Train character-level language model

✅ Implement and understand attention mechanism

⏳ Add model checkpoint saving and loading

⏳ Integrate multi-head + feedforward improvements

⏳ Generate syntax-valid Python functions, classes, docstrings

📜 License
This project uses open-source data and is provided under the MIT License. Please ensure any generated code complies with original licenses if redistributed.

🙌 Credits
Inspired by Andrej Karpathy’s nanoGPT series
GitHub repositories used: TheAlgorithms/Python and python/cpython

---

Let me know if you'd like me to add:
- Code examples (for usage in a `Usage` section)
- Badges (e.g., Python version, license, etc.)
- Docker or virtualenv setup instructions

This README is already detailed and production-quality, but happy to tailor it further!
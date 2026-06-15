# Lab 01 — Development Environment Setup

**NAVTTC Task 1 | Week 1**

Complete this checklist on your machine and keep screenshots for your portfolio/viva.

## Required Tools

| Tool | Purpose | Install Command / Link |
|------|---------|------------------------|
| Python 3.10+ | Core language | https://www.python.org/downloads/ |
| Anaconda3 | Data science environment | https://www.anaconda.com/download |
| VS Code | Primary IDE | https://code.visualstudio.com/ |
| PyCharm (optional) | Python IDE | https://www.jetbrains.com/pycharm/download/ |
| Git | Version control | https://git-scm.com/downloads |

## Python Packages (run after Anaconda install)

```bash
conda create -n navttc-ai python=3.11 -y
conda activate navttc-ai

pip install numpy pandas matplotlib seaborn scikit-learn xgboost
pip install tensorflow torch torchvision
pip install nltk gensim jupyter
```

## Verify Installation

```bash
python --version
python -c "import numpy, pandas, sklearn, tensorflow, torch; print('All core libraries OK')"
jupyter notebook --version
```

## Evidence for GitHub / Viva

- [ ] Screenshot: `python --version` output
- [ ] Screenshot: VS Code with this repo open
- [ ] Screenshot: Jupyter notebook running `navttc_practice_workbook.ipynb`
- [ ] Screenshot: `import tensorflow as tf; print(tf.__version__)`
- [ ] Screenshot: `import torch; print(torch.__version__)`

## Notes

- TensorFlow 2.x and PyTorch are both required per NAVTTC syllabus.
- Use `navttc-ai` conda environment for all labs in this repository.
- GPU is optional; all labs run on CPU.

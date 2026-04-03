# Project instructions

## Main references

- Jin, Peng, Xitong Zhang, Yinpeng Chen, Sharon Xiaolei Huang, Zicheng Liu, and Youzuo Lin. 2021. “Unsupervised Learning of Full-Waveform Inversion: Connecting CNN and Partial Differential Equation in a Loop.” http://arxiv.org/abs/2110.07584.


---

## Development instructions

This project can be developed either using **Google Colab** or on your **local machine**.

### Choose your development environment

- **Google Colab** → for quick start and minimal setup  
- **Local machine** (recommended) → for more control and advanced development  

In all cases, make sure you are working on **your own branch** and commit your changes regularly.  
---

## 1. Google Colab

If you prefer a quick setup without installing anything locally, use **Google Colab**.

Organize your notebook as follows:

- **First cell**: repository setup (Git + clone + navigation)  
- **Second cell**: install project dependencies  
- **Last cell**: commit and push your changes  

---

### First cell: repository setup

It is assumed that each student has already created their **own working branch** following the instructions in the root repository (`piml-pub/README.md`).

```python
# Configure Git (needed once per Colab session)
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

# Clone your own branch
!git clone -b your-branch-name --single-branch https://github.com/brunoss-on/piml-pub.git

# Enter the repository directory
%cd piml-pub

# List files
!ls
````

---

### Second cell: install project dependencies

Example for projects using `deepwave`:

```python
# Install project-specific dependencies
!pip install deepwave==0.0.22
```

---

### Continue your development

After setup, proceed normally with your notebook.

---

### Last cell: saving your work (commit and push)

```python
# Ensure you are inside the repository
%cd /content/piml-pub

# Check changes
!git status

# Add changes
!git add .

# Commit
!git commit -m "Update notebook"

# Push to your branch
!git push origin your-branch-name
```

---

## 2. Local development

These instructions assume that **Conda is already installed** on your computer.

If Conda is not installed, install **Miniconda** using the official guide:
[https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)

---

### Why use a dedicated environment?

* isolates project dependencies from your system
* avoids version conflicts with other projects
* improves reproducibility across different machines

---

### Environment setup

Inside **your branch** and within the project subdirectory, define an `environment.yml` file with the project dependencies.

- Keep all project dependencies updated and documented in the `environment.yml` file


Example for projects using `deepwave`:

```yaml
name: deepwave-lab
channels:
  - conda-forge
  - defaults

dependencies:
  # Core
  - python=3.10
  - pip

  # Essentials
  - numpy
  - scipy
  - matplotlib
  - pandas
  - pyyaml
  - pillow
  - tqdm
  - ipython
  - jupyterlab
  - ipykernel
  - ipywidgets

  # Notebook tools
  - nbdime

  # Seismic tools
  - segyio
  - obspy

  - pip:
      # PyTorch
      - torch==2.8.*
      - torchvision==0.23.*

      # Deepwave
      - deepwave==0.0.22

      # Utilities
      - sympy
      - torchsummary
```

---

### Optional: Makefile for environment management 

Use the following `Makefile` to create, update, or remove the Conda environment.  
Remember to replace `ENV_NAME ?= deepwave-lab` with the name of the environment defined in your `environment.yml` file.

```makefile
SHELL := /bin/bash
ENV_FILE ?= environment.yml
ENV_NAME ?= deepwave-lab

.PHONY: env-create env-update env-remove

env-create:
	conda env create -f $(ENV_FILE)

env-update:
	conda env update -f $(ENV_FILE) --prune

env-remove:
	conda env remove -n $(ENV_NAME)
```

---

### Usage

```bash
make env-create
conda activate deepwave-lab
make env-update
make env-remove
```

---

## Recommended day-to-day git workflow

After your branch has already been created, the usual workflow is:

```bash
git checkout your-branch-name
git pull
git status
git add .
git commit -m "Describe your changes"
git push
```



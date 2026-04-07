# Project Guidelines

## Overview

This project focuses on the development and application of computational methods in geophysics, with an emphasis on reproducible workflows and scientific programming.


## Main Reference

  - Alfarhan, Mustafa, Matteo Ravasi, Fuqiang Chen, and Tariq Alkhalifah. 2024. “Robust Full Waveform Inversion with Deep Hessian Deblurring.” Geophysical Journal International 240 (1): 303–16. https://doi.org/10.1093/gji/ggae378.

---

## Development Environment

You may choose one of the following:

- **Google Colab** → recommended for quick start and zero setup  
- **Local machine (recommended)** → better control and reproducibility  

In all cases:

- Work **only on your own branch**
- Commit your progress regularly
- Keep your code organized and reproducible

---

## 1. Google Colab

Use this option if you prefer not to install anything locally.

For consistency and reproducibility, **organize your notebook with the following structure**:

- **First cell** → repository setup (Git configuration and clone)  
- **Second cell** → dependency installation  
- Remaining cells → development and experiments  
- Final cell → commit and push your changes  

---

### First cell: repository setup

```python
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

!git clone -b your-branch-name --single-branch https://github.com/your-repo-url.git

%cd your-repo-name
!ls
````

---

### Second cell: install dependencies

> Replace with project-specific dependencies if needed.

```python
!pip install deepwave==0.0.22
```

---

### Development

* Organize your code clearly
* Document your steps
* Structure your experiments

---

### Final cell: save and push your work

```python
%cd /content/your-repo-name

!git status
!git add .
!git commit -m "Update notebook"
!git push origin your-branch-name
```

---

## 2. Local Development (Recommended)

### Requirements

* Conda (Miniconda or Anaconda)

---

### Why use a dedicated environment?

* isolates dependencies
* avoids version conflicts
* ensures reproducibility

---

### Environment setup

Inside **your branch** and within the project subdirectory, define an `environment.yml` file with the project dependencies.

> Keep all project dependencies updated and documented in the `environment.yml` file
> Adapt dependencies according to the project.

Example:

```yaml
name: project-env
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

### Makefile for environment management (Optional)

Use the following `Makefile` to create, update, or remove the Conda environment.  

```makefile
SHELL := /bin/bash
ENV_FILE ?= environment.yml
ENV_NAME ?= project-env

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
conda activate project-env
make env-update
make env-remove
```

---

## Git Workflow

```bash
git checkout your-branch-name
git pull
git status
git add .
git commit -m "Describe your changes"
git push
```

---

## Good Practices

* Keep notebooks clean and well-structured
* Avoid committing unnecessary files
* Document experiments and assumptions
* Use meaningful commit messages
* Keep dependencies updated

---

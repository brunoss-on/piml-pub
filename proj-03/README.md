# Project Guidelines

## Overview

This project focuses on the development and application of computational methods in geophysics, with an emphasis on reproducible workflows and scientific programming.


## Main Reference

- Jin, P., Zhang, X., Chen, Y., Huang, S. X., Liu, Z., and Lin, Y. 2021. “Unsupervised Learning of Full-Waveform Inversion: Connecting CNN and Partial Differential Equation in a Loop.” arXiv:2110.07584.
---


## Development Environment

You may choose one of the following:

- **Google Colab** 
- **Local machine (recommended)** → better control and reproducibility  

In all cases:

- Work **only on your own branch**
- Commit your progress regularly
- Keep your code organized and reproducible

---

## 1. Google Colab (Simplified Workflow)

### Step 1. Create your branch on GitHub

Before opening Colab:

1. Access the repository on GitHub
2. Create a new branch for your work
3. Use a clear and standardized naming convention:

```text
student-name/proj-number
```

**Example:**

```text
bruno/proj-01
```

---

### Step 2. Open the repository in Colab

1. Go to Google Colab
2. Select the **GitHub** tab
3. Search for the repository
4. Choose **your branch** (not `main`)
5. Open the notebook directly

---

### Step 3. Work inside the notebook

To ensure **consistency and reproducibility**, follow this structure:

* **First cell → Environment setup**

  * Install all dependencies (`pip install`, etc.)
  * Configure any required settings

* **Second cell → Data access**

  * Ensure all required datasets are available
  * Recommended options:

    * Google Drive (for large datasets)
    * Repository files (for small datasets)

* **Remaining cells → Development**

  * Keep code organized and modular
  * Clearly document your steps
  * Structure experiments in a logical sequence

---

### Step 4. Save and commit your work

When finishing your session:

1. In Colab, go to **File → Save**
2. Select:

   * The  **repository**
   * Your **branch**
   * The notebook file to update
3. Add a clear and descriptive **commit message**
4. Enable the option **Include a link to colab**

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

# Project Guidelines

## Overview

This project focuses on the development and application of computational methods in geophysics, with an emphasis on reproducible workflows and scientific programming.


## Main Reference

  - Rasht-Behesht, M., Huber, C., Shukla, K., and Karniadakis, G. E. 2022. “Physics-Informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions.” Journal of Geophysical Research: Solid Earth 127 (5). https://doi.org/10.1029/2021JB023120

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

## Architectural Documentation: PINN for FWI (Charles Lima's notes)
Reference: Rasht-Behesht, M., Huber, C., Shukla, K., and Karniadakis, G. E. (2022).

1. What We Adopted (The Theoretical Core)
Our implementation perfectly preserves the mathematical foundation proposed by the authors:

Coordinate-Based Parameterization: Instead of updating a discrete grid of velocity values directly (as in classical FWI), we use a Multilayer Perceptron (MLP) to map continuous spatial coordinates (x, z) to the velocity field v(x, z). The neural network itself acts as the velocity model.

Physics-Informed Regularization: We are utilizing the PyTorch Autograd engine to compute the spatial derivatives. This allows the network to be constrained not just by the seismic data, but by the physical laws governing acoustic wave propagation.

Composite Loss Function: Our optimization process follows the paper's dual-objective approach, simultaneously minimizing:
Loss = Loss_data + (lambda * Loss_PDE)

2. What We Adjusted (The Hybrid Escalation)
While the paper demonstrates the PINN's ability to solve both the forward and inverse problems purely through neural networks via collocation points, we introduced a strategic engineering adjustment to handle the specific bottleneck of high-frequency data and multiple shots:

The Deepwave Integration: Pure PINNs struggle with spectral bias and massive memory consumption when calculating the transient wavefield u(t, x, z) over thousands of time steps. To solve this, our architecture is a Hybrid Data-Driven PINN (DDR-PINN).

The Division of Labor: We use the PINN to parameterize and optimize the velocity macro-model v(x, z). However, we offload the heavy lifting of the time-domain wave propagation (the forward problem to compute Loss_data) to Deepwave, which uses highly optimized finite-difference solvers.

The Justification: This adjustment maintains the exact physical constraints proposed by Rasht-Behesht et al. while allowing our code to scale to the 5-shot OpenFWI geometry without immediately crashing standard hardware.

Code Documentation Standards: Traceability & Reference Tracking
To ensure scientific rigor and transparent methodology, all codebase developments within this repository employ a mandatory dual-tagging system within the inline comments. This system explicitly maps our engineering choices to the primary theoretical framework:

[Rasht-Behesht 2022 Compliance]: Indicates where the implementation strictly follows the original mathematical or theoretical foundation proposed by the authors.

[Rasht-Behesht 2022 Adaptation]: Indicates where we intentionally diverge from the original paper to optimize for High-Performance Computing (HPC), manage VRAM memory bottlenecks, or ensure numerical stability.

Inversion Strategy: Step 1 (Acoustic Forward Stress Test)The first phase of this repository validates the forward acoustic wave propagation and backpropagation (autograd) mechanics before introducing the complex optimization loops required for FWI.

To ensure mathematical rigor and justify our architectural choices, we have implemented two parallel baselines for Step 1. This ablation study compares the theoretical ideal against a computationally optimized approach.

1. The Theoretical Baseline (pure_pinn_baseline.ipynb)

* Methodology: 100% compliant with the original methodology proposed by Rasht-Behesht et al. (2022).
  
* Architecture: Utilizes two separate continuous Neural Networks (MLPs): one to approximate the transient wavefield u(t, x, z) and another for the velocity model v(x, z).
  
* Data Ingestion: Integrates a vectorized PyTorch data loader to directly ingest the original SPECFEM2D training data (wavefield snapshots and surface seismograms) provided by the authors, ensuring an identical ground-truth comparison.
  
* Purpose: To validate the elegance of pure Physics-Informed Neural Networks where the acoustic wave equation is solved purely by penalizing the PDE residual via PyTorch Automatic Differentiation (Autograd) on random collocation points.

2. The HPC-Optimized Evolution (pinn_inversion_step1.ipynb)

* Methodology: A Hybrid Data-Driven PINN (DDR-PINN).

* Architecture: Explicitly diverges from the pure PINN by offloading the continuous wavefield approximation to Deepwave (a PyTorch-native finite-difference solver). It retains the dense discrete grid v(x,z) which acts as the optimizable tensor, with requires_grad=True forcing the GPU to track the full computational graph.

* Hardware Scaling: Implements pure, unadulterated tensor parallelization, processing the entire multi-source array (5+ shots) simultaneously on the GPU's Streaming Multiprocessors.

* Purpose: To solve the severe computational and spectral bias bottlenecks of pure PINNs. By using Deepwave for the forward pass, we bypass the need for millions of collocation points, enabling the processing of high-frequency seismic data at scale without crashing local VRAM limits.
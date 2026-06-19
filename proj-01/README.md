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

**Reference:** Rasht-Behesht, M., Huber, C., Shukla, K., and Karniadakis, G. E. (2022). *Physics-Informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions.*

### 1. Theoretical Core & Hybrid Escalation (DDR-PINN)
Our implementation preserves the coordinate-based parameterization of the original authors, utilizing a Multilayer Perceptron (MLP) to map spatial coordinates (x, z) to the velocity field v(x, z). However, to scale this foundational work to real-world, high-frequency, multi-source seismic domains, we engineered the **Hybrid Data-Driven PINN (DDR-PINN)**. By delegating the computationally massive time-domain wave propagation to Deepwave (a highly optimized finite-difference solver), we bypass the severe VRAM bottlenecks and exponential computational scaling required by pure continuous PINNs, maintaining the physics-informed gradients while enabling industrial HPC viability.

* **[Rasht-Behesht 2022 Compliance]:** The optimization simultaneously minimizes data mismatch and physics residuals.
* **[Rasht-Behesht 2022 Adaptation]:** The forward problem is offloaded from PyTorch collocation points to Deepwave's finite-difference engines.

---

### 2. Phase 1: Forward Acoustic Wave Propagation & Benchmarking
The first phase validated the mechanics of the DDR-PINN against a pure PINN baseline (`pure_pinn_baseline.ipynb`). When constrained to practical compute timeframes, the Pure PINN exhibited severe **spectral bias**—successfully mapping low-frequency global trends but failing to resolve the high-frequency acoustic wavefield. The DDR-PINN successfully bypassed this bottleneck, maintaining stable convergence and memory efficiency for complex multi-shot geometries.

---

### 3. Phase 2: FWI & The Cycle Skipping Limitation
Building upon the hybrid architecture, Phase 2 executed the closed-loop Full Waveform Inversion. The model attempted to invert a structural anomaly (2500 m/s) from a blind, homogeneous background (1500 m/s) using high-frequency 25Hz synthetic data. 

**The Local Minimum Trap:**
Despite a massive reduction in MSE loss, the network failed to reconstruct the subsurface anomaly. Because the 25Hz source generates short acoustic wavelengths, the time delay between the true and predicted wavefields exceeded a half-wavelength. The optimizer suffered from **cycle skipping**, converging into a local minimum by making microscopic, non-physical adjustments to the background velocity to falsely align incorrect wave peaks.

---

### 4. Phase 3: The Latent-Fourier & TV Annealing Evolution
To achieve true quantitative interpretation, the DDR-PINN architecture was fundamentally upgraded to resolve both the kinematic cycle skipping of Phase 2 and the inherent spectral bias of coordinate-based MLPs.

1. **Multi-Scale Frequency Sweeping:** Implemented a sequential inversion pipeline ($5\text{Hz} \rightarrow 15\text{Hz} \rightarrow 25\text{Hz}$) with dynamic *Optimizer Kicks*. Broad wavelengths map the macro-model first, physically preventing cycle skipping in the high-frequency passes.
2. **Latent-Fourier Feature Mapping:** Replaced raw $(x,z)$ coordinates with high-dimensional sinusoidal projections. This destroyed the spatial spectral bias, granting the network the mathematical capacity to draw sharp, orthogonal impedance contrasts.
3. **Total Variation (TV) Annealing:** Injected a dynamic topological penalty to compensate for deep subsurface shadow zones. TV forces macro-block formation at 5Hz but gracefully decays to zero at 25Hz, preventing the fatal binarization of high-frequency migration artifacts.

---

### 5. Activity 6: Quantitative Ablation Study (DDR-PINN vs. Classical FWI)
**Objective:** To benchmark the structural resolution capabilities of the fully upgraded Latent-Fourier DDR-PINN against a deterministic Full-Waveform Inversion (FWI) baseline, using the industry-standard L-BFGS optimizer.

**Experimental Constraints (The Sparse Regime):**
Both architectures were subjected to a severely ill-posed physical environment:
* **Initial State:** Blind, homogeneous 1500 m/s background.
* **Acquisition:** Extreme sparse-data constraint (5 surface shots).
* **Target:** A 2500 m/s high-velocity structural anomaly.
* **Evaluation:** Metrics extracted via a strict Binary Masking protocol targeting only the anomaly zone to prevent background inflation.

**Quantitative Results:**
| Architecture | Mean Squared Error (MSE) | Structural Similarity Index (SSIM) | Visual Status |
| :--- | :--- | :--- | :--- |
| **Classical FWI (L-BFGS)** | 300,000.00 | 0.5748 | Total Algorithmic Collapse (Failed to update from 1500 m/s) |
| **Hybrid DDR-PINN** | 1,099.89 | 0.9815 | Successful Target Reconstruction (Orthogonal geometry secured) |

**Conclusion:**
Under dense acquisition arrays, classical FWI is the industry standard. However, this ablation study proves that under extreme sparse-data regimes, classical L-BFGS suffers catastrophic failure due to insufficient gradient illumination. The DDR-PINN circumvents this physical limitation by leveraging Latent-Fourier mapping and TV Annealing to hallucinate the missing physics and recover the structural target.

---

### 6. Epilogue: The Validity of the Ablation Dispute
A critical distinction must be made regarding the nature of the failures encountered during this R&D cycle to validate the fairness of comparing a neural architecture against classical FWI.

* **The Classical FWI Failure (Physical Limit):** The L-BFGS optimizer collapsed due to **Gradient Starvation**. With only 5 surface shots, the deterministic equations mathematically cannot update the grid because the physical wavefield provides no information in the deep shadow zones. This is an insurmountable limit of sparse data acquisition.
* **The Original PINN Failure (Architectural Limit):** Standard coordinate-based MLPs fail to draw sharp boundaries due to **Spatial Spectral Bias**. They are mathematically constrained to learning smooth representations. 

**Conclusion on Fairness:**
This ablation study is mathematically sound because both algorithms were subjected to the exact same physical constraints. It proves that while classical deterministic inversion dies in sparse-data regimes, the hybrid DDR-PINN architecture survives. The neural network acts as an advanced non-linear regularizer, reconstructing the missing physics through geometric logic and topological constraints.
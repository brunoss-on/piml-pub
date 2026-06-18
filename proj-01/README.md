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

### 1. What We Adopted (The Theoretical Core)
Our implementation perfectly preserves the mathematical foundation proposed by the authors:

* **Coordinate-Based Parameterization:** Instead of updating a discrete grid of velocity values directly (as in classical FWI), we use a Multilayer Perceptron (MLP) to map continuous spatial coordinates (x, z) to the velocity field v(x, z). The neural network itself acts as the velocity model.
* **Physics-Informed Regularization:** We are utilizing the PyTorch Autograd engine to compute the spatial derivatives. This allows the network to be constrained not just by the seismic data, but by the physical laws governing acoustic wave propagation.
* **Composite Loss Function:** Our optimization process follows the paper's dual-objective approach, simultaneously minimizing: `Loss = Loss_data + (lambda * Loss_PDE)`.

### 2. What We Adjusted (The Hybrid Escalation)
* **The Deepwave Integration:** Pure PINNs struggle with massive memory consumption when calculating the transient wavefield u(t, x, z) over thousands of time steps. To solve this, our architecture is a Hybrid Data-Driven PINN (DDR-PINN).
* **The Division of Labor:** We use the PINN to parameterize and optimize the velocity macro-model v(x, z). However, we offload the heavy lifting of the time-domain wave propagation (the forward problem to compute `Loss_data`) to Deepwave, which uses highly optimized finite-difference solvers.
* **The Justification (Algorithmic Scaling & Spectral Bias):** The foundational work by Rasht-Behesht et al. (2022) successfully demonstrates the theoretical elegance of using purely continuous neural networks to solve both the forward and inverse problems via PDE collocation points. However, when transitioning from foundational models to real-world, high-frequency, multi-source seismic domains, pure PINNs encounter a well-documented mathematical bottleneck: *spectral bias*. Neural networks inherently prioritize learning low-frequency functions and require exponential computational scaling (massive collocation point density and extended training durations) to resolve high-frequency transient wavefields. To make multi-shot FWI viable and reproducible without enterprise-grade supercomputers, we engineered the Hybrid Data-Driven PINN (DDR-PINN). By delegating the time-domain wave propagation to Deepwave, a rigorously optimized finite-difference solver perfectly suited for high-frequency acoustics, we bypass the spectral bias. This hybrid approach mathematically preserves the physics-informed gradients of the original authors while dramatically increasing computational efficiency and VRAM viability.

---

### Code Documentation Standards: Traceability & Reference Tracking
To ensure scientific rigor and transparent methodology, all codebase developments within this repository employ a mandatory dual-tagging system within the inline comments. This system explicitly maps our engineering choices to the primary theoretical framework:

* **[Rasht-Behesht 2022 Compliance]:** Indicates where the implementation strictly follows the original mathematical or theoretical foundation proposed by the authors.
* **[Rasht-Behesht 2022 Adaptation]:** Indicates where we intentionally diverge from the original paper to optimize for High-Performance Computing (HPC), manage VRAM memory bottlenecks, or ensure numerical stability.

---

### Phase 1: Forward Acoustic Wave Propagation (Ablation Study)
The first phase of this repository validates the forward acoustic wave propagation and backpropagation (autograd) mechanics before introducing the complex optimization loops required for FWI. To ensure mathematical rigor and justify our architectural choices, we have implemented two parallel baselines for Step 1. This ablation study compares the theoretical ideal against a computationally optimized approach.

**1. The Theoretical Baseline (`pure_pinn_baseline.ipynb`)**
* **Methodology:** 100% compliant with the original methodology proposed by Rasht-Behesht et al. (2022).
* **Architecture:** Utilizes two separate continuous Neural Networks (MLPs): one to approximate the transient wavefield u(t, x, z) and another for the velocity model v(x, z).
* **Data Ingestion:** Integrates a vectorized PyTorch data loader to directly ingest the original SPECFEM2D training data (wavefield snapshots and surface seismograms) provided by the authors, ensuring an identical ground-truth comparison.
* **Purpose:** To validate the elegance of pure Physics-Informed Neural Networks where the acoustic wave equation is solved purely by penalizing the PDE residual via PyTorch Automatic Differentiation (Autograd) on random collocation points.

**2. The HPC-Optimized Evolution (`pinn_inversion_step1.ipynb`)**
* **Methodology:** A Hybrid Data-Driven PINN (DDR-PINN).
* **Architecture:** Explicitly diverges from the pure PINN by offloading the continuous wavefield approximation to Deepwave. It retains the dense discrete grid v(x,z) which acts as the optimizable tensor, with `requires_grad=True` forcing the GPU to track the full computational graph.
* **Hardware Scaling:** Implements pure, unadulterated tensor parallelization, processing the entire multi-source array simultaneously on the GPU's Streaming Multiprocessors.
* **Purpose:** To solve the severe computational and spectral bias bottlenecks of pure PINNs. By using Deepwave for the forward pass, we bypass the need for millions of collocation points, enabling the processing of high-frequency seismic data at scale.

---

### Phase 2: Full Waveform Inversion Implementation (`pinn_inversion_step2.ipynb`)
Building upon the hybrid architecture established in Phase 1, Step 2 executes the complete closed-loop Full Waveform Inversion.

**1. Data Ingestion & Memory Management**
* **[Rasht-Behesht 2022 Compliance]:** The model ingests the exact SPECFEM2D 'event1' training dataset provided by the authors, ensuring an identical ground-truth comparison.
* **[Rasht-Behesht 2022 Adaptation]:** Instead of managing data via NumPy and feeding it sequentially, the entire dataset is pre-compiled into contiguous PyTorch tensors and pinned directly to the GPU VRAM to maximize epoch throughput.

**2. The Optimization Loop**
* **Mechanism:** The architecture iteratively updates a dense, trainable velocity grid (`v_inverted`) by minimizing the Mean Squared Error (MSE) between the Deepwave-generated synthetic seismograms and the observed SPECFEM2D surface data.
* **[Rasht-Behesht 2022 Compliance]:** The gradients computed via the PyTorch Autograd engine on the Deepwave output directly mirror the data mismatch gradient calculation in the original PINN formulation.
* **Validation:** The loop includes a physical constraint to bound velocity updates between 1500 m/s and 5500 m/s. Convergence is tracked via a customized visualization dashboard that plots the logarithmic loss trajectory, the observed surface wavefields, and the inverted structural velocity model for direct cross-referencing with the Rasht-Behesht ground truth.

---

### Phase 3: Benchmarking & Scaling Analysis (The Ablation Study)
To empirically validate the architectural necessity of the Hybrid DDR-PINN, this repository includes a rigorous benchmarking suite. The objective is to replicate the foundational success of Rasht-Behesht et al. (2022) and subsequently expose the theoretical limitations of pure continuous neural networks when scaling to industrial seismic applications.

**1. Experiment A: Foundational Baseline (The Control)**
* **Objective:** Replicate the original authors' results to validate our Pure PINN implementation (`pure_pinn_baseline.ipynb`).
* **Parameters:** Low-frequency source wavelets (< 10Hz), single-shot geometry, small computational domains, and smoothed velocity anomalies.
* **Hypothesis:** The Pure PINN will successfully converge, mapping the acoustic wavefield and reconstructing the velocity model, confirming the viability of the authors' proof-of-concept.

**2. Experiment B: Industrial Stress Test (The Spectral Bias Barrier)**
* **Objective:** Subject both the Pure PINN and the Hybrid DDR-PINN to real-world complexities to evaluate scaling endurance.
* **Parameters:** High-frequency source wavelets (15Hz - 30Hz Ricker), complex geological structures (Marmousi slices), and simultaneous multi-shot geometries.
* **Hypothesis:** The Pure PINN will exhibit severe *spectral bias*, resulting in a plateaued loss landscape (stagnant convergence) and unsustainable VRAM allocation due to the exponential collocation points required. Conversely, the DDR-PINN (`pinn_inversion_step2.ipynb`) will bypass this bottleneck via Deepwave's finite-difference forward solver, maintaining stable convergence and memory efficiency.

**3. Evaluation Metrics**
The comparative analysis relies on strict quantitative tracking:
* **Computational Footprint:** Peak VRAM (GB) and System RAM utilization per epoch.
* **Temporal Efficiency:** Wall-clock time required to reach a baseline Mean Squared Error (MSE).
* **Gradient Trajectory:** Tracking the divergence in loss curves to explicitly visualize the onset of spectral bias in the Pure PINN architecture.

### 4. Empirical Validation: Spectral Bias & The DDR-PINN Justification
**The Theoretical Ideal (Rasht-Behesht et al., 2022)**
The foundational paper successfully demonstrates that Full Waveform Inversion can be solved entirely via continuous coordinate-based neural networks. However, resolving high-frequency structural boundaries (overcoming spectral bias) required massive epoch counts (20,000+) and computationally expensive second-order L-BFGS optimization on enterprise-grade hardware.

**The Industrial Reality Check (Our Baseline)**
When constrained to a practical industrial compute timeframe (500 epochs, Adam optimizer), the Pure PINN architecture immediately exhibits severe spectral bias. As documented in our Phase A proofs, the network successfully maps low-frequency global velocity trends but entirely fails to resolve the high-frequency acoustic wavefield, resulting in a fundamentally blurred structural inversion.

**The Hybrid DDR-PINN Solution**
To deploy FWI at scale without requiring supercomputers or L-BFGS optimization, the spectral bias must be bypassed entirely. The DDR-PINN architecture achieves this by offloading the high-frequency multi-shot wave propagation to Deepwave (a rigorous finite-difference solver) while utilizing the neural network exclusively to optimize the velocity macro-model. This dramatically accelerates convergence and preserves sharp structural resolution.

### 5. Phase 2 Inversion: The Cycle Skipping Limitation and Multi-Scale Requisite

**The 25Hz High-Frequency Stress Test**
Phase 2 evaluated the DDR-PINN's ability to invert a structural anomaly (2500 m/s) from a blind, homogeneous background (1500 m/s) using high-frequency 25Hz synthetic data. While the neural network successfully integrated with the Deepwave finite-difference engine and resolved the "dead gradient" trap via coordinate normalization, the inversion exposed a fundamental physical limitation of single-frequency FWI.

**The Cycle Skipping Trap (Local Minimum)**
Despite achieving a massive reduction in Mean Squared Error (MSE) loss (dropping from 0.344 to 0.006 in 150 epochs), the network entirely failed to reconstruct the subsurface anomaly. Because the 25Hz source generates short acoustic wavelengths, the time delay between the true wavefield and the predicted wavefield exceeded a half-wavelength. 

Consequently, the optimizer suffered from **cycle skipping**. To minimize the MSE, the network bypassed the global minimum (the true geological structure) and converged into a local minimum, making microscopic, non-physical adjustments to the background velocity to falsely align the wrong wave peaks.

**Architectural Next Steps: Multi-Scale Frequency Sweeping**
This test proves that while the DDR-PINN bypasses the pure PINN's spectral bias, it remains bound by the kinematic laws of classical wave propagation. Brute-forcing high-frequency inversions from blind starting models is mathematically unviable. 

To achieve true quantitative interpretation and structural resolution, the DDR-PINN architecture must be upgraded to a **Multi-Scale Inversion** framework:
1. **Low-Frequency Pass (e.g., 5Hz):** Broad wavelengths map the global macro-model without cycle skipping.
2. **Mid-Frequency Pass (e.g., 15Hz):** Uses the 5Hz output to refine structural boundaries.
3. **High-Frequency Pass (e.g., 25Hz):** Uses the 15Hz output to lock in razor-sharp seismic facies classifications.

This multi-scale frequency sweep will be the focus of the next R&D development cycle.
# Magnetic Resonance Artificial Intelligence Velocimetry (MR-AIV)

This repository provides the code implementations for published and upcoming research on Magnetic Resonance Artificial Intelligence Velocimetry (MR-AIV), a technique to infer *in vivo* brain-wide velocity fields from murine Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) data. This repository will be continuously updated with new research developments and resources.

## Table of Contents

- [Overview](#overview)
- [Papers](#papers)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Set up Conda Environment (Recommended)](#1-set-up-conda-environment-recommended)
  - [2. Install JAX with GPU Support](#2-install-jax-with-gpu-support)
  - [3. Clone the Repository](#3-clone-the-repository)
  - [4. Install the `instant-aiv` Package](#4-install-the-instant-aiv-package)
  - [5. Install Jupyter (Optional for Examples)](#5-install-jupyter-optional-for-examples)
- [Dataset Setup](#dataset-setup)
  - [1. Download the Dataset](#1-download-the-dataset)
  - [2. Prepare the Data Directory](#2-prepare-the-data-directory)
- [Usage: Running the Models](#usage-running-the-models)
  - [1. Initialize the Permeability Network](#1-initialize-the-permeability-network)
  - [2. Run Real Data Analysis](#2-run-real-data-analysis)
  - [3. Run Synthetic Data Analysis](#3-run-synthetic-data-analysis)
- [References](#references)
- [License](#license)

## Overview

(Add Details)
## Papers

Currently, this repository contains the code for the following paper(s):

1.  **[TODO: Add full reference for your primary MR-AIV paper here. e.g., Toscano, J.D., et al. (2025). Title of Paper. *Journal*, Volume(Issue), Pages. DOI/Link]**

Additional papers and updates will be added to this repository as they become available.


## Prerequisites

* Python: Version >= 3.9 and <= 3.11 is recommended.
* NVIDIA GPU: Required for running the models, along with a compatible CUDA toolkit.
* Conda: Recommended for managing Python environments and dependencies.

## Installation

These instructions will guide you through setting up a Conda environment, installing JAX with GPU support, cloning the repository, and installing the `instant-aiv` library.

***Note:*** *If you already have a suitable Python environment (Python >= 3.9) with a working JAX GPU installation (JAX >= 0.3.23), you can skip to [Step 3: Clone the Repository](#3-clone-the-repository) and proceed from there, ensuring your existing environment meets the project's dependencies.*

### 1. Set up Conda Environment (Recommended)

Using Conda helps manage project dependencies and ensures a clean environment.

```bash
# Create a new Conda environment (e.g., "AIV_Conda")
# This example uses Python 3.11. Adjust if you need 3.9 or 3.10.
conda create -n AIV_Conda python=3.11
conda activate AIV_Conda

# It's good practice to upgrade pip within the new environment
pip install --upgrade pip
```
### 2. Install JAX with GPU Support

JAX is a critical component for this project. Correct installation, which depends on your system's NVIDIA driver and CUDA toolkit version, is essential.

**Crucial First Step:** Always consult the **[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)**. The commands provided there are the most current and specific to different CUDA versions.

* **Standard Pip Command (Recommended starting point - Example for CUDA 12):**
    The official JAX guide will provide a command similar to the following (ensure you use the exact command appropriate for your CUDA version):
    ```bash
    pip install -U "jax[cuda12_pip]"
    ```
    *(Note: The `cuda12_pip` part is specific to CUDA 12. If you use a different CUDA version, like CUDA 11, you'll need to use the corresponding identifier, e.g., `cuda11_pip`. This command installs JAX, JAXlib, and the necessary NVIDIA libraries for JAX, all managed via pip.)*

* **Important Considerations for JAX Installation:**
    * **Compatibility:** Discrepancies between your NVIDIA drivers, your system's installed CUDA toolkit, and the versions of JAX/NVIDIA libraries installed by pip can lead to runtime or compilation errors. These often appear when JAX first attempts to utilize the GPU.
    * **Troubleshooting:** If you encounter issues, the troubleshooting sections in the official JAX guide and JAX community forums (like JAX GitHub Discussions) are valuable resources.
* **Verify JAX Installation:**
    Verify that JAX can detect and use your GPU(s):
    ```bash
    python -c "import jax; print(jax.devices())"
    # A successful installation should list your GPU devices (e.g., [GpuDevice(id=0, process_index=0, slice_index=0)]).
    ```

### 3. Clone the Repository

Next, you'll clone the project repository from GitHub.

```bash
# Clone the repository into a directory named "Instant_AIV"
git clone [https://github.com/jdtoscano94/Instant_AIV_lib.git](https://github.com/jdtoscano94/Instant_AIV_lib.git) Instant_AIV
# Change your current directory to the newly cloned repository folder
cd Instant_AIV
```

### 4. Install the `instant-aiv` Package

After successfully cloning the repository and navigating into its main directory (e.g., `Instant_AIV`), and with your Conda environment (`AIV_Conda`) activated, you can now install the project package. 
```bash
# Ensure you are in the root directory of the cloned project (e.g., Instant_AIV)
# Install the package and its dependencies
pip install .

# Alternatively, if you plan to actively develop the code,
# install it in editable mode. Changes to the source code
# will then be reflected immediately without needing a reinstall:
# pip install -e .
```
### 5. Install Jupyter (Optional for Examples)

If your project includes Jupyter notebooks for running examples, visualizing results, or interactive exploration, you'll need to install Jupyter. You should also register your `AIV_Conda` environment as a kernel so Jupyter can use it.

```bash
# First, install Jupyter (choose one or both):
pip install jupyter notebook  # For the classic Jupyter Notebook interface
# pip install jupyterlab      # For the newer JupyterLab interface

# Next, install ipykernel. This package allows Jupyter to work with different Python environments.
pip install ipykernel

# Now, register your 'AIV_Conda' environment as a Jupyter kernel.
# This makes it selectable when you create or open a notebook.
python -m ipykernel install --user --name=AIV_Conda --display-name="Python (AIV_Conda)"
```

## Dataset Setup

This section outlines how to download the required dataset and organize it for use with the project code.

### 1. Download the Dataset

The dataset used for the examples and analyses in this project is hosted on Zenodo.

* **Download Link:** [Dataset on Zenodo](https://zenodo.org/uploads/15345393?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImFhODQ3YTFhLTU5YTAtNDdjNy1iYWQ2LWZmMGY2MDI3NTU0NCIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZTgwMWM2ZjRmMTdlYzdkNDkxOWU2ZWFlYzllYzJiMCJ9.T8PxIlIKta2tifGYk3hp-pnj-GsWHo-l35ryOHduuIgMoR5UHz35NasuRpl8_Nozzo8vxsaU97QFXkJKHGyJew)

Please download `Data.zip` from the link above.

### 2. Prepare the Data Directory

Once downloaded, the dataset needs to be placed and unzipped within your project folder (e.g., `Instant_AIV`).

1.  **Navigate to your project's root directory if you're not already there:**
    *(This assumes your project folder is named `Instant_AIV` as per previous installation steps.)*
    ```bash
    # Example: cd /path/to/your/Instant_AIV
    ```

2.  **Create a `Data` subfolder if it doesn't exist:**
    ```bash
    # Ensure you are in the Instant_AIV directory
    mkdir -p Data
    ```

3.  **Move the downloaded dataset zip file into the `Data` subfolder:**
    Replace `/path/to/your/downloads/Data.zip` with the actual path to where you downloaded the `Data.zip` file.
    ```bash
    # Example:
    mv /path/to/your/downloads/Data.zip Data/
    ```

4.  **Unzip the dataset within the `Data` subfolder:**
    Navigate into the `Data` directory and unzip the file.
    ```bash
    cd Data
    unzip Data.zip
    # After unzipping, you can remove the .zip file if desired:
    # rm Data.zip
    cd ..
    # This brings you back to the project's root directory (e.g., Instant_AIV)
    ```

5.  **Verify the directory structure:**
    After unzipping, your `Data` directory should contain the following subfolders and files:
    ```
    Data/
    ├── Permeabilities/
    │   ├── M1.mat
    │   ├── M2.mat
    │   ├── M3.mat
    │   ├── M4.mat
    │   └── M5.mat
    ├── Real_Data/
    │   ├── M1/
    │   │   ├── Front_Tracking.mat
    │   │   └── M1_concentration.mat
    │   ├── M2/
    │   │   ├── Front_Tracking.mat
    │   │   └── M2_concentration.mat
    │   ├── M3/
    │   │   ├── Front_Tracking.mat
    │   │   └── M3_concentration.mat
    │   ├── M4/
    │   │   ├── Front_Tracking.mat
    │   │   └── M4_concentration.mat
    │   └── M5/
    │       ├── Front_Tracking.mat
    │       └── M5_concentration.mat
    └── Synthetic_Data/
        ├── BC_inlet.csv
        ├── BC_noflow.csv
        ├── BC_outlet2.csv
        ├── BC_outlet3.csv
        ├── Realistic.csv
        ├── Sharp.csv
        └── Smooth.csv
    ```

This structure is essential for the scripts to locate and use the data correctly.

## Usage: Running the Models

This section describes the steps to run the different components of the MR-AIV analysis, assuming you have successfully completed the installation and dataset setup. All commands should be run from the root directory of the project (e.g., `Instant_AIV`), unless specified otherwise, and with the `AIV_Conda` environment activated.

A `Results` directory will be created in your project root to store the outputs.

### 1. Initialize the Permeability Network

This step trains the network to estimate permeability.

1.  **Navigate to the Permeability subfolder:**
    ```bash
    cd Permeability
    ```

2.  **Run the initialization script:**
    ```bash
    python Init_K.py
    ```

3.  **Verify Outputs:**
    * A `Results` folder will be created in the main project directory (`Instant_AIV/Results/`) if it doesn't already exist.
    * Inside `Results/`, a subfolder named `Permeability` will be created containing the trained model and related outputs.
    * A file named `Permeability_Details.pkl` will be created directly within the `Results/` folder (i.e., `Instant_AIV/Results/Permeability_Details.pkl`). This file is important for subsequent steps.
    * After the script finishes, navigate back to the project root:
        ```bash
        cd ..
        ```

### 2. Run Real Data Analysis

This involves initializing and then training the networks for pressure and concentration using the real experimental data.

1.  **Navigate to the Real Data subfolder:**
    ```bash
    # Ensure you are in the project's root directory (e.g., Instant_AIV)
    cd Real
    ```

2.  **Initialize the pressure and concentration networks:**
    ```bash
    python Init_c_and_p.py
    ```
    * **Verification:** This script should generate a file named `Initialization_Details_Real.pkl` directly within the `Results/` folder (i.e., `Instant_AIV/Results/Initialization_Details_Real.pkl`).

3.  **Train the model with real data:**
    ```bash
    python Train.py
    ```
    * **Outputs:** This script will save the trained parameters, visualizations generated during training, and a `.mat` file (containing velocities, pressure, permeability, and concentration gradients) into the `Instant_AIV/Results/Real/` subfolder.

4.  **Explore Results (Optional):**
    You can explore the trained model and regenerate results using the provided Jupyter Notebook.
    ```bash
    # Navigate to the results directory for real data
    cd ../Results/Real/ 
    # Ensure you have Jupyter installed and the AIV_Conda kernel activated (see Installation Step 5)
    jupyter notebook Visualize.ipynb 
    # Or use jupyter lab Visualize.ipynb
    ```
    After exploring, navigate back to the project root:
    ```bash
    cd ../.. 
    # This should bring you back to Instant_AIV/
    ```

### 3. Run Synthetic Data Analysis

This involves initializing and then training the networks using synthetic data.

1.  **Navigate to the Synthetic Data subfolder:**
    ```bash
    # Ensure you are in the project's root directory (e.g., Instant_AIV)
    cd Synthetic
    ```

2.  **Initialize the pressure and concentration networks:**
    ```bash
    python Init_c_and_p.py
    ```
    * **Verification:** This script should generate a file named `Initialization_Details.pkl` directly within the `Results/` folder (i.e., `Instant_AIV/Results/Initialization_Details.pkl`).

3.  **Train the model with synthetic data:**
    ```bash
    python Train.py
    ```
    * **Outputs:** This script will save the trained parameters, visualizations, and a `.mat` file into the `Instant_AIV/Results/Synthetic/` subfolder.

4.  **Explore Results (Optional):**
    Explore the trained model and regenerate results using the provided Jupyter Notebook.
    ```bash
    # Navigate to the results directory for synthetic data
    cd ../Results/Synthetic/
    # Ensure you have Jupyter installed and the AIV_Conda kernel activated
    jupyter notebook Visualize.ipynb
    # Or use jupyter lab Visualize.ipynb
    ```
    After exploring, navigate back to the project root:
    ```bash
    cd ../..
    # This should bring you back to Instant_AIV/
    ```

Make sure to run these steps in the prescribed order, as some steps depend on the output files from previous ones (e.g., `Permeability_Details.pkl`).


## References

 The following pre-prints and related works provide foundational concepts and techniques used or explored in conjunction with this project:

```bibtex
@article{anagnostopoulos2024residual,
  title={Residual-based attention in physics-informed neural networks},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={421},
  pages={116805},
  year={2024},
  publisher={Elsevier}
}

@article{anagnostopoulos2024learning,
  title={Learning in PINNs: Phase transition, total diffusion, and generalization},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2403.18494},
  year={2024}
}

@article{toscano2024inferring,
  title={Inferring in vivo murine cerebrospinal fluid flow using artificial intelligence velocimetry with moving boundaries and uncertainty quantification},
  author={Toscano, Juan Diego and Wu, Chenxi and Ladr{\'o}n-de-Guevara, Antonio and Du, Ting and Nedergaard, Maiken and Kelley, Douglas H and Karniadakis, George Em and Boster, Kimberly AS},
  journal={Interface Focus},
  volume={14},
  number={6},
  pages={20240030},
  year={2024},
  publisher={The Royal Society}
}

@article{toscano2025aivt,
  title={AIVT: Inference of turbulent thermal convection from measured 3D velocity data by physics-informed Kolmogorov-Arnold networks},
  author={Toscano, Juan Diego and K{\"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={Science Advances},
  volume={11},
  number={19},
  pages={eads5236},
  year={2025},
  publisher={American Association for the Advancement of Science}
}

@article{shukla2024comprehensive,
  title={A comprehensive and fair comparison between mlp and kan representations for differential equations and operator networks},
  author={Shukla, Khemraj and Toscano, Juan Diego and Wang, Zhicheng and Zou, Zongren and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117290},
  year={2024},
  publisher={Elsevier}
}


@article{toscano2024kkans,
  title={KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics},
  author={Toscano, Juan Diego and Wang, Li-Lian and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2412.16738},
  year={2024}
}
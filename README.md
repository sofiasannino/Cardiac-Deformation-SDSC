<div align="center">

# Spatio-temporal Probabilistic Model for Cardiac Deformation

### A 4D Cardiac Deformation Model with Control-Point Gaussian Splatting and Temporal Gaussian Process Priors

Project carried out at the Swiss Data Science Center, under the supervision of Dr. Quentin Duchemin and Dr. David Bruggeman.

<br>

<img width="429" height="117" alt="sdsc_logo" src="https://github.com/user-attachments/assets/d4c9bec7-d000-4f87-9d53-dc402c5a0560" />

[Project Report](https://sofiasannino.github.io/files/Semester_project_Sannino_SDSC.pdf) ·
[Author](https://github.com/sofiasannino.github.io)

</div>

---

## Overview

This project investigates an interpretable framework for learning cardiac deformation from **4D cine-MRI**.

The proposed pipeline combines:

- a geometry-based algorithm for extracting anatomically consistent control points;
- a 3D coordinate-regression network;
- temporal Gaussian Process priors for smoothing cardiac trajectories;
- Gaussian Splatting for differentiable volumetric reconstruction.

The experimental work focuses on the extraction of pseudo-ground-truth control points and the pre-training of the coordinate-regression network using the **Automated Cardiac Diagnosis Challenge**, or **ACDC**, dataset.

> [!NOTE]
> The control-point extraction and coordinate-regression stages were implemented and experimentally evaluated. The Gaussian Process and Gaussian Splatting components provide the theoretical basis for future development of the complete end-to-end pipeline.

---

## Model

```text
4D cine-MRI sequence
        │
        ▼
Coordinate-regression network
        │
        ▼
Predicted control points
        │
        ▼
Temporal Gaussian Process
        │
        ▼
Smoothed control-point trajectories
        │
        ▼
Gaussian Splatting
        │
        ▼
Reconstructed cardiac volumes
```

For each MRI frame, a 3D U-Net encoder followed by a coordinate-regression head predicts a sparse collection of cardiac control points.

The predicted coordinates are treated as noisy observations of latent cardiac trajectories. A temporal Gaussian Process is introduced to obtain smoother and more physiologically plausible motion across the cardiac cycle.

Finally, the corrected control points can be converted into a dense Gaussian representation and rendered back into image space using Gaussian Splatting.

---

## Repository structure

```text
Cardiac-Deformation-SDSC/
├── README.md
├── requirements.txt
│
├── explore_dimensions.py       # Explore MRI volume dimensions
├── inference.py                # Run model inference and compute metrics
├── preprocess.py               # Convert ACDC data to the nnU-Net format
├── pretrain.py                 # Interpolate frames and prepare pre-training data
├── pretrain_unet.py            # Train the coordinate-regression U-Net
├── train.py                    # Extract and visualize cardiac control points
├── unet_infer.py               # Run U-Net coordinate inference
│
└── src/
    ├── configs/                # Hydra experiment and model configurations
    ├── datasets/               # Dataset classes, loading, and data utilities
    ├── logger/                 # Experiment logging utilities
    ├── loss/                   # Training loss functions
    ├── metrics/                # Evaluation metrics
    ├── model/                  # Control-point and neural-network models
    ├── trainer/                # Training, inference, and plotting logic
    ├── transforms/             # Image and coordinate transformations
    └── utils/                  # Initialization, I/O, and shared utilities
```

The repository uses **Hydra** and **OmegaConf** for configuration management. Training, inference, preprocessing, and data-analysis scripts load their settings from `src/configs/`. 

The main entry points are:

| Script | Purpose |
|---|---|
| `preprocess.py` | Converts ACDC images and labels into the nnU-Net directory format and creates five patient-wise data splits. |
| `pretrain.py` | Resamples cardiac cycles to a fixed temporal length while preserving ED and ES, runs intermediate-frame segmentation, and reorganizes the results. |
| `pretrain_unet.py` | Trains the coordinate-regression U-Net and plots its losses. |
| `unet_infer.py` | Runs coordinate inference and plots test losses. |
| `train.py` | Extracts control points from a segmentation mask and visualizes the resulting 2D and 3D points. |
| `inference.py` | Instantiates the configured model, dataloaders, transforms, and metrics and then saves predictions. |
| `explore_dimensions.py` | Calculates MRI dimension statistics and saves histograms of depth, height, and width. |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sofiasannino/Cardiac-Deformation-SDSC.git
cd Cardiac-Deformation-SDSC
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> The command above assumes that the repository contains a `requirements.txt` file.

---

## Data availability
The ACDC dataset is not distributed with this repository.
It must be downloaded separately from the official challenge website and used according to its terms and conditions.

---


## Project information

This work was completed as a semester project in the **Section of Mathematics at EPFL**, in collaboration with the **Swiss Data Science Center**.

Author: Sofia Sannino  
Responsible professor: Prof. Olivier Verscheure  
Supervisors: David Brüggemann and Quentin Duchemin  
Period: Spring 2026  
---

## Citation

If you use or build upon this work, please cite it as:

```bibtex
@misc{sannino2026cardiac,
  author       = {Sofia Sannino},
  title        = {A 4D Cardiac Deformation Model with Control-Point
                  Gaussian Splatting and Temporal Gaussian Process Priors},
  year         = {2026},
  institution  = {EPFL and Swiss Data Science Center},
  note         = {Semester Project}
}
```


<div align="center">


</div>

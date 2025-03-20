# Advanced Indoor Pathloss Prediction Using Deep Learning

## Authors
- **Sajid Javid** (PHD24002)
- **Nikhil Suri** (2021268)
- **Siddhant Gautam** (2021100)

## Overview
This project addresses the critical task of indoor pathloss prediction, essential for the design and optimization of wireless indoor networks. We leverage deep learning methodologies to overcome the limitations of traditional empirical models, enhancing prediction accuracy and generalization across various indoor environments.

## Objectives
- Develop deep learning models to accurately predict indoor pathloss.
- Evaluate and ensure model generalization to unseen indoor geometries, frequencies, and antenna radiation patterns.

## Tasks Breakdown

### Task 1: Geometry Generalization
- **Simulation Parameters:** Isotropic antenna (Ant1) at 868 MHz (f1)
- **Data:** Buildings B1-B25, 50 radio maps each
- **Objective:** Validate model generalization on 5 unseen building geometries.

### Task 2: Frequency and Geometry Generalization
- **Simulation Parameters:** Isotropic antenna (Ant1) at 0.868, 2, and 3.5 GHz (f1, f2, f3)
- **Data:** Buildings B1-B25, 50 radio maps per building per frequency
- **Objective:** Validate generalization on unseen geometries and an unseen frequency band.

### Task 3: Comprehensive Generalization
- **Simulation Parameters:** 5 antenna patterns (Ant1-Ant5) at 0.868, 2, and 3.5 GHz
- **Data:** 
  - Ant1: 50 radio maps per building per frequency
  - Ant2-Ant5: 80 radio maps per building per frequency (random steering angles)
- **Objective:** Validate model generalization across new geometries, frequencies, and antenna patterns.

## Dataset Structure
- RGB channels representing Transmittance (G), Reflectance (R), and Transmitter Location (B).
- Novel feature channels:
  - **Line of Sight (LoS) Channel:** Computed from input RGB channels and frequency.
  - **Antenna Radiation Pattern:** Incorporates geometric and electromagnetic transformations.

## Model Architectures
- **UNet:** Encoder-decoder architecture ideal for spatial feature extraction.
- **WNet:** Dual-path encoder network for capturing complex structural features.
- **DeepLabV3:** Atrous convolutions and group normalization, optimal for semantic segmentation.

## Methodology
### Data Preprocessing
- Image resizing (256x256 pixels)
- Pixel normalization ([0, 1] scale)

### Data Augmentation
- Vertical/horizontal flipping
- Random rotations (0°, 90°, 180°, 270°)
- Synchronized transformations for input-output consistency
- Building-wise Randomization (BR) for enhanced model robustness

### Training Configuration
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam optimizer (learning rate: 0.0001 to 0.001)
- **Epochs:** 200 epochs with early stopping
- **Batch Size:** 16

## Experiments and Results
- Demonstrated improved accuracy with advanced features (LoS channel, antenna radiation pattern).
- Enhanced performance observed with frequency-specific scaling (lambda scaling).
- Comprehensive experiments across geometries, frequencies, and antennas confirming robust model generalization.

## Conclusions
- Significant improvements in indoor pathloss prediction through advanced deep learning techniques.
- Novel feature integration (LoS and radiation pattern channels) substantially improved accuracy.
- Proven effectiveness of advanced architectures (UNet, WNet, DeepLabV3) and data augmentation strategies.

## References
1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015.
2. Chen et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets," IEEE TPAMI, 2017.
3. ICASSP 2024 Dataset and Challenge Documentation.

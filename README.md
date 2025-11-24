# Uncertainty-Aware Deep Learning for Turbofan Engine RUL Prediction

A novel deep learning framework for Remaining Useful Life (RUL) prediction in turbofan engines that learns aleatoric uncertainty directly through probabilistic modeling, achieving breakthrough critical zone performance with RMSE of 5-7 cycles for RUL ≤ 30.

## Overview

This framework addresses critical limitations in the current literature by combining hierarchical deep learning with Bayesian uncertainty quantification. The architecture synergistically integrates multi-scale Inception blocks for temporal pattern extraction, bidirectional LSTM for sequential modeling, and a novel dual-level attention mechanism operating simultaneously on sensor and temporal dimensions. The innovation lies in the Bayesian output layer that predicts both mean RUL and variance, enabling the model to learn data-inherent uncertainty, an approach unexplored in existing CMAPSS literature.

### Dataset

The pipeline is evaluated on all four NASA CMAPSS benchmark subsets which depict turbofan engine degradation under varying operational conditions. FD001/FD003 represent single operating condition scenarios with 100 engines each, while FD002/FD004 represent complex multi-condition scenarios with 249-260 engines across six operating regimes and multiple fault modes.

## Features

- Breakthrough Critical Zone Performance — Achieves 5-7 cycle RMSE for RUL ≤ 30, representing 25-40% improvement over existing methods
- Learned Uncertainty Quantification — Bayesian output layer provides well-calibrated 95% confidence intervals with 93.5-95.2% actual coverage
- Dual-Level Attention — Simultaneous sensor and temporal attention for comprehensive feature extraction
- Multi-Scale Temporal Modeling — Inception blocks with parallel convolutions capture patterns at multiple timescales
- Condition-Aware Processing — K-means clustering and regime-specific normalization for multi-modal operating conditions
- Intelligent Preprocessing — Wavelet denoising achieving 15-25 dB SNR improvement and correlation-based feature selection

## Key Innovations

- Dual-Level Attention Mechanism — Our novel attention operates simultaneously on sensor and temporal dimensions with learnable fusion weights, enabling the model to dynamically identify which sensors are relevant at each degradation state and which time steps are most predictive.
- Bayesian Uncertainty Quantification — The output layer simultaneously predicts mean RUL and log-variance, learning when predictions should be uncertain versus confident. This enables maintenance planners to make risk-aware decisions scheduling immediate inspections for high-uncertainty critical predictions.
- RUL-Aware Loss Weighting — Critical samples receive 2.5× higher weight during training, concentrating model capacity on safety-critical zones. This design philosophy recognizes that prediction errors are not equally consequential.
- Condition-Aware Processing — K-means clustering on operational settings enables regime-specific normalization, preventing cross-regime interference where identical sensor values may indicate different degradation states under different flight conditions.

## Comparison with State-of-the-Art

While recent methods working with the CMAPSS dataset like TMSCNN achieve lower overall RMSE (14.79, 14.25) on complex subsets of the CMAPSS dataset and transformers excel on the simpler subsets (11.36 RMSE), our framework offers unique capabilities unavailable in existing approaches: learned uncertainty quantification with calibrated confidence intervals, breakthrough critical zone accuracy where safety matters most, and computational efficiency suitable for real-time edge deployment. 

## Performance

Our framework achieves competitive overall RMSE of 16.22, 19.29, 16.84, and 19.98 on FD001-FD004 respectively. Most significantly, new benchmarks are established for safety-critical predictions with critical zone RMSE (RUL ≤ 30) of 5.14, 6.89, 5.27, and 7.16 cycles, representing 25-40% improvements over conventional approaches. The learned uncertainty provides well-calibrated confidence intervals enabling risk-aware maintenance scheduling previously unattainable in CMAPSS literature.

## How To Run

1. Clone this repository on your local machine, it has the standard CMAPSS dataset as a zipped file (extract it in the project directory), and the Jupyter Notebook code file.
2. Install the required dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib pywavelets ruptures
```
3. Open and execute the Jupyter Notebook `CMAPSS RUL Prediction.ipynb` which contains the complete end-to-end pipeline from data loading through training to evaluation and visualization. By default, in the main method, the data subset is set to FD001, which can be easily changed to FD002, FD003, or FD004.

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 

# Invariant Theory vs. GNNs: An Empirical Comparison

**A Project for EN.553.743: Equivariant Machine Learning**

## Introduction
This project provides an empirical comparison between Invariant Theory-based models and Graph Neural Networks (GNNs) on two key tasks: molecular property prediction and point cloud distance prediction. 

## Report
The detailed project report can be found in the `report/` directory.

## Code Structure
The repository is organized into two main directories, each corresponding to a distinct experiment:

-   `molecules/`: Contains all source code and experiment setup for the task of molecular property prediction.
-   `pointclouds/`: Contains all source code and experiment setup for the task of point cloud distance prediction.


## Datasets
The datasets used in this project are consistent with those in the paper ["A Galois theorem for machine learning: Functions on symmetric matrices and point clouds via lightweight invariant features"](https://arxiv.org/pdf/2405.08097).

-   **Molecular Property Prediction**: We use the **QM7b** dataset, which is available through the PyTorch Geometric library. It can be loaded directly using `torch_geometric.datasets.QM7b`.
-   **Point Cloud Distance Prediction**: We use the **ModelNet40** dataset. Following the methodology of [nhuang37/InvariantFeatures](https://github.com/nhuang37/InvariantFeatures), we sub-sample each point cloud to 100 points for processing.

## Getting Started

### Prerequisites

Ensure you have a Python environment set up. The primary dependencies for this project follow the requirements of the [nhuang37/InvariantFeatures](https://github.com/nhuang37/InvariantFeatures) repository.

You can install the necessary packages, including PyTorch and PyTorch Geometric, by following their official installation guides.

### Running Experiments

To run the experiments for either task, navigate to the corresponding subdirectory (`molecules/` or `pointclouds/`) and simply execute `python main.py`.


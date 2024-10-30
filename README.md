# Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks

This is the README file for the project [Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks](https://www.sciencedirect.com/science/article/pii/S0169207024000967), published in [International Journal of Forecasting](https://www.sciencedirect.com/journal/international-journal-of-forecasting). 
It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)

## Project Structure

The project is organized as follows:

- `data_subsample.py`: Subsample the minutely data to 5 minutes and merge the data of all stocks in the stock list.
- `compute_vol.py`: Compute the daily variance from 5-min return data.
- `GHAR.py`: Linear models to forecast the realized volatility, including HAR and GHAR. HAR is a special case of GHAR, assuming the adjacency matrix is identity.
- `GNNHAR.py`: Proposed GNNHAR models to forecast the realized volatility. 
- `MCS.py`: Implementation of Econometrica Paper: "The model confidence set." by Hansen, Peter R., Asger Lunde, and James M. Nason. 
- `Summary_Results.py`: Summarize the results of the forecast models, including the MSE, QLIKE, and the MCS tests.
- `Summary_Regime.py`: Summarize the results of the forecast models, based on different regimes.
- `BoxPlot_Error.py`: Plot the boxplot of the forecast error and ratio for different models

## Usage

To use the project, follow these steps:

1. Download LOBSTER data (minutely or higher freq) and save to your local path
2. Run data_subsample.py and compute_vol.py sequentially
3. Run GHAR.py to obtain the baseline forecasts from linear regressions
4. Run GNNHAR.py to obtain the forecasts for proposed GNNHAR models
5. Compare their forecasts by using Summary_Results.py and Summary_Regime.py
6. Generate plots by BoxPlot_Error.py


## Data
The data used in this reproducibility check is LOBSTER (https://lobsterdata.com/), which needs to be purchased by users.

## Computing Environment
To run the reproducibility check, the following computing environment and package(s) are required:
- Environment: These experiments were conducted on a system equipped with an Nvidia A100 GPU with 40 GB of GPU memory, an AMD EPYC 7713 64-Core Processor @ 1.80GHz with 128 cores, and 1.0TB of RAM, running Ubuntu 20.04.4 LTS. 

- Package(s): 
    - Python 3.8.18
    - PyTorch 2.0.1+cu117
    - numpy 1.22.3
    - pandas 2.0.3
    - scikit-learn 1.3.0
    - matplotlib 3.7.2

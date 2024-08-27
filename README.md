# Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks

This is the README file for the project "Forecasting Realized Volatility with Spillover Effects: Perspectives from Graph Neural Networks". It provides an overview of the project structure and instructions on how to use the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)

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

1. Download LOBSTER data (minutely or higher freq) from https://lobsterdata.com/ and save to your local path
2. Run data_subsample.py and compute_vol.py sequentially
3. Run GHAR.py to obtain the baseline forecasts from linear regressions
4. Run GNNHAR.py to obtain the forecasts for proposed GNNHAR models
5. Compare their forecasts by using Summary_Results.py and Summary_Regime.py. 
6. Generate plots by BoxPlot_Error.py

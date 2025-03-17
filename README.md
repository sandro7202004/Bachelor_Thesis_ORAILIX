# Bachelor Thesis Internship: Train Delay Prediction

This repository contains the code and scripts used for my Bachelor Thesis on **Train Delay Prediction**. The dataset is not available in this repository due to privacy concerns and the large size of the files.

## Repository Structure

The repository is structured as follows:

- `utils.py`: Utility functions for the whole project.
- `newfeatures.ipynb`: Notebook for feature creation.
- `one_month/`:
  - `best_radii.ipynb`: Notebook for determining the best radii for feature selection.
  - `data_visualization.ipynb`: Visualizations and data analysis.
  - `PCA.ipynb`: Analysis for selecting the best PCA components.
  - A folder for XGBoost model training.
  - A folder for XGBoost model training with PCA.
  - A folder for MLP (Multi-Layer Perceptron) model training.
  - A `not_selected` folder for other models that were tested but not selected.
- `full_year/`:
  - `data_visualization.ipynb`: Data visualization and analysis for full-year dataset.
  - `gridsearch_training.ipynb`: GridSearch for full year progressive study.
  - A folder for XGBoost model training for full-year data.
  - A folder for MLP model training for full-year data.

## Project Overview

This project explores various machine learning techniques to predict train delays based on historical data. The models are trained on different time spans:

- **One Month Data:**
  - XGBoost with and without PCA.
  - PCA Component Analysis.
  - MLP Model.
  - Other models (not selected).
  - Data Visualization.
  - Best Radii Determination.

- **Full Year Data:**
  - XGBoost.
  - MLP.
  - GridSearch for progressive study.
  - Data Visualization.

## Results & Analysis

- The models were evaluated only based on **Mean Absolute Error (MAE)**.
- The best performing model was determined based on MAE among other factors (refer to report for more details).

## Contact
For any questions, feel free to reach out:

- **Name:** Alessandro Sayad
- **Email:** alessandro.sayad@polytechnique.edu

---

*This repository is part of my Bachelor Thesis in Train Delay Prediction. The dataset is not included due to privacy concerns and file size limitations.*

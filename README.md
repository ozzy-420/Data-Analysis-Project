# Data Analysis Project

## Overview
This project focuses on performing data analysis using Python. It includes various scripts for  visualization, and statistical analysis.

## Features
- Data Visualization
- Statistical Analysis

## Installation
To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/ozzy-420/Data-Analysis-Project.git
cd Data-Analysis-Project
pip install -r requirements.txt
```
# How to run the program?

To run the prgram you must run the data_analysis_main.py
If you want to run it on the defeault setting then make sure that you have downloaded [this](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv) data and that you put it in the data folder in the project under the name "UCI_Credit_Card.csv", as it is used as defeault.

## Parameters
data_source: Path to the input data file (e.g., data/UCI_Credit_Card.csv).
output_dir: (Optional) Path to the folder where results will be saved. By default, results are saved in the output/{file_name}_key_visuals folder.
plot_config: (Optional) Configuration for the plots, such as DPI and color palette.
mapping: (Optional) A dictionary for mapping categorical values to their labels (e.g., changing numeric values to text in the SEX and EDUCATION columns).

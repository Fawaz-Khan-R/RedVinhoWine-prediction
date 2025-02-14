# Wine Quality Prediction Project

## Overview

This project aims to predict the quality of Vinho Verde wines based on their physicochemical properties. The dataset includes both red and white wine variants, and this project explores machine learning techniques to build a predictive model for the red kind.

## Dataset Information

The dataset used in this project is the "Wine Quality" dataset, which contains information about red and white variants of the Portuguese "Vinho Verde" wine. The dataset includes physicochemical properties of the wines as well as sensory data (quality scores). Due to privacy and logistical reasons, only physicochemical (inputs) and sensory (the output) variables are available (e.g., there is no data about grape types, wine brand, wine selling price, etc.).

*   **Source:** The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) but originally from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).
*   **Size:** The dataset contains 6,497 instances (wines), including 4,898 white wines and 1,599 red wines.
*   **Type:** The dataset includes both red and white wine variants.

### Features

The dataset includes the following features:

*   **Fixed Acidity:** The amount of tartaric acid fixed in the wine.
*   **Volatile Acidity:** The amount of acetic acid in the wine, which can lead to an unpleasant vinegar taste if too high.
*   **Citric Acid:** The amount of citric acid in the wine, which can add a refreshing flavor.
*   **Residual Sugar:** The amount of sugar remaining in the wine after fermentation.
*   **Chlorides:** The amount of salt in the wine.
*   **Free Sulfur Dioxide:** The amount of free sulfur dioxide in the wine, which acts as an antioxidant and antibacterial agent.
*   **Total Sulfur Dioxide:** The total amount of sulfur dioxide in the wine, including both free and bound forms.
*   **Density:** The density of the wine.
*   **pH:** The acidity level of the wine.
*   **Sulphates:** The amount of sulfates in the wine, which can contribute to a bitter taste.
*   **Alcohol:** The alcohol content of the wine.
*   **Type:** Indicates whether the wine is red or white.
*   **Quality:** The sensory score of the wine, ranging from 0 to 10 (the output variable).

## Methodology

The following steps were taken in this project:

1.  **Data Loading:** The dataset was loaded into a pandas DataFrame.
2.  **Data Exploration:** Exploratory data analysis (EDA) was performed to understand the distribution of the features and identify any missing values.
3.  **Data Cleaning:** Missing values were handled by imputing them with appropriate values.
4.  **Data Preprocessing:** Encoding categorical features to numerical values.
5.  **Model Training:** A machine-learning model was trained on the dataset to predict wine quality.
6.  **Model Evaluation:** The trained model was evaluated on a testing dataset to assess its accuracy and performance.
7.  **Model Optimization:** Hyperparameter tuning was performed to improve the model's performance.

## Installation

To run this project, you will need to install the following dependencies:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
plt.style.use('ggplot')

import plotly.express as px
```

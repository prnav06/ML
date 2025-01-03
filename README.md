# Coronary Heart Disease Prediction

This project focuses on predicting coronary heart disease (CHD) using logistic regression and decision tree models. The dataset used for this project contains records of males from a high-risk heart disease region in the Western Cape, South Africa. The aim is to analyze the dataset and evaluate the performance of the two models.

## Dataset Overview

The dataset contains information from a study described in **Rousseauw et al. (1983), South African Medical Journal**. It is a tab-separated file (`.csv`) containing the following columns:

- `sbp`: Systolic blood pressure
- `tobacco`: Cumulative tobacco consumption (kg)
- `ldl`: Low-density lipoprotein cholesterol
- `adiposity`: Body fat percentage
- `famhist`: Family history of heart disease (`Present` or `Absent`)
- `typea`: Type-A behavior score
- `obesity`: Obesity measurement
- `alcohol`: Current alcohol consumption
- `age`: Age at onset
- `chd`: Response variable, indicating coronary heart disease (`1` for positive, `0` for negative)

### Notes on the Dataset:
- The dataset contains roughly two controls per CHD case.
- Many CHD-positive individuals underwent treatments (e.g., blood pressure reduction) after their CHD event, which might influence some measurements.
- These data were extracted from a larger dataset.

## Project Goals
1. Explore and preprocess the dataset.
2. Train and evaluate logistic regression and decision tree models for CHD prediction.
3. Compare the models based on performance metrics.

## Methodology
### Data Preprocessing
- Categorical variables (e.g., `famhist`) were encoded using one-hot encoding.
- Features were analyzed for significance based on p-values from logistic regression.

### Models
1. **Logistic Regression**:
   - Performed feature selection to include only statistically significant variables.
   - Used Youden's Index to determine the optimal cutoff probability for predictions.

2. **Decision Tree**:
   - Performed hyperparameter tuning using GridSearchCV.
   - Evaluated the model's performance using the best hyperparameters.

## Key Results
- **Performance Metrics**:
  - **Logistic Regression**:
    - Optimized using Youden's Index and cost-based analysis.
    - Achieved the best overall performance among the models.
  - **Decision Tree**:
    - Tuned using `criterion` and `max_depth` parameters.
    - Demonstrated competitive results but slightly lower performance than logistic regression.

- **Comparison**:
  Logistic regression outperformed the decision tree in terms of ROC-AUC score, particularly when using Youden's Index for probability threshold optimization.

## Dependencies
The following Python libraries were used:
- `numpy`
- `pandas`
- `statsmodels`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Project Structure

    SA Heart.csv: Dataset used for the project.
    CHD.py: Jupyter Notebook containing the code and analysis.
    README.md: Documentation for the project.

## References

    Rousseauw et al. (1983). South African Medical Journal.


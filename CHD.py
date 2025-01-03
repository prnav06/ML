# %%
# %%

# Importing essential libraries
import numpy as np
import pandas as pd

# %%
# %%

# Reading the dataset of South Africa Medical Journal
chd_df = pd.read_csv("SA Heart.csv")

# %%
# %%

# Displaying the dataset information for an overview of its structure
chd_df.info()

# %%
# %%

# Extracting feature names to predict the target variable 'chd'
X_features = list(chd_df.columns)
X_features.remove('chd')  # Removing the target variable from the feature list

# %%
# %%

# Splitting the dataset into features (X) and target variable (Y)
X = chd_df[X_features]  # Independent variables
Y = chd_df.chd          # Dependent variable (target)

# %%
# %%

# Encoding the categorical variable 'famhist' using one-hot encoding
encoded_X = pd.get_dummies(X, drop_first=True)

# %%
# %%

# Checking the datatype of encoded features; boolean data needs conversion
encoded_X.info()

# %%
# %%

# Converting 'bool' data type to 'int64' for all encoded columns
encoded_X.iloc[:, 9:] = encoded_X.iloc[:, 9:].astype('int64')

# %%
# %%

# Importing the train-test split function from scikit-learn
from sklearn.model_selection import train_test_split

# %%
# %%

# Splitting the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    encoded_X,
    Y,
    train_size=0.7,
    random_state=100
)

# %%
# %%

# Importing the statsmodels library for logistic regression
import statsmodels.api as sm

# %%
# %%

# Fitting the initial logistic regression model with all features
logit_model_1 = sm.Logit(y_train, sm.add_constant(X_train)).fit()

# %%
# %%

# Displaying the summary of the logistic regression model
logit_model_1.summary2()

# %%
# %%

# Function to extract significant variables based on p-values (threshold: 0.05)
def get_signficant_vars(lm):
    var_p_vals_df = pd.DataFrame(lm.pvalues, columns=['pvals'])
    var_p_vals_df['vars'] = var_p_vals_df.index
    
    # Returning variables with p-values less than or equal to 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05].vars)

# %%
# %%

# Extracting significant variables from the initial model
significant_vars = get_signficant_vars(logit_model_1)
significant_vars.remove('const')  # Removing the constant term
significant_vars

# %%
# %%

# Building a refined logistic regression model using only significant variables
logit_model_2 = sm.Logit(y_train, sm.add_constant(X_train[significant_vars])).fit()

# %%
# %%

# Displaying the summary of the refined logistic regression model
logit_model_2.summary2()

# %%
# %%

# Creating a DataFrame to store actual and predicted values
y_pred_df = pd.DataFrame()
y_pred_df['actual'] = y_test
y_pred_df['pred'] = logit_model_2.predict(sm.add_constant(X_test[significant_vars]))
y_pred_df

# %%
# %%

# Importing Seaborn for data visualization
import seaborn as sns
from sklearn import metrics

# %%
# %%

# Function to calculate the optimal cutoff probability using Youden's Index
def find_optimal_cutoff_probability(actual, pred):
    idx = 0
    cut_off_df = pd.DataFrame(columns=['cut_off_prob', 'diff_tpr_fpr'])
    
    for cp in range(10, 51):
        cp_pred = pred.map(lambda x: 1 if x > (cp / 100) else 0)

        conf_matr = metrics.confusion_matrix(actual, cp_pred)

        tp = conf_matr[0][0]
        fp = conf_matr[1][0]
        fn = conf_matr[0][1]
        tn = conf_matr[1][1]

        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate

        cut_off_df.loc[idx] = [cp / 100, tpr - fpr]  # Youden's Index

        idx += 1
        cp += 1

    cut_off_df = cut_off_df.sort_values('diff_tpr_fpr', ascending=False)

    return cut_off_df.iloc[0, 0]  # Returning the optimal cutoff probability

# %%
# %%

# Finding the optimal cutoff probability using Youden's Index
optimal_cutoff_probability = find_optimal_cutoff_probability(y_pred_df.actual, y_pred_df.pred)
optimal_cutoff_probability

# %%
# %%

# Applying the optimal cutoff probability to generate final predictions
y_pred_df['optimal_pred'] = y_pred_df.pred.map(
    lambda x: 1 if x > optimal_cutoff_probability else 0
)

# %%
# %%

# Function to compute and visualize the confusion matrix
import matplotlib.pyplot as plt

def get_cm(actual, optimal_pred):
    conf_matr_cutoff_prob = metrics.confusion_matrix(actual,
                                                     optimal_pred,
                                                     labels=[1, 0])

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matr_cutoff_prob, annot=True, fmt='.2f',
                xticklabels=['chd_positive', 'chd_negative'],
                yticklabels=['chd_positive', 'chd_negative'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# %%
# %%

# Displaying the confusion matrix for the predictions
get_cm(y_pred_df.actual, y_pred_df.optimal_pred)

# %%
# %%

# Generating a classification report for logistic regression using Youden's Index
classification_report_1 = metrics.classification_report(y_pred_df.actual,
                                                        y_pred_df.optimal_pred)
print(classification_report_1)

# %%
# %%

# Function to compute the optimal cutoff probability using cost-based analysis
def get_optimal_cost_cutoff(actual, pred, cost_FN, cost_FP):
    def get_total_cost(actual, pred, cost_FN, cost_FP):
        cm = metrics.confusion_matrix(actual, pred, labels=[1, 0])
        return cost_FP * cm[1][0] + cost_FN * cm[0][1]  # Total cost calculation

    idx = 0
    cost_df = pd.DataFrame(columns=['cut_off_prob', 'cost'])

    for cp in range(10, 51):
        cost_df.loc[idx] = [(cp / 100), get_total_cost(
            actual, pred.map(lambda x: 1 if x > (cp / 100) else 0),
            cost_FN, cost_FP)]

        idx += 1
        cp += 1

    cost_df = cost_df.sort_values('cost')

    return cost_df.iloc[0, 0]  # Returning the optimal cutoff probability

# %%
# %%

# Finding the optimal cutoff probability using cost-based analysis
optimal_cost_cutoff_probability = get_optimal_cost_cutoff(y_pred_df.actual, y_pred_df.pred, 5, 1)
optimal_cost_cutoff_probability

# %%
# %%

# Applying cost-based optimal cutoff probability to generate predictions
y_pred_df['optimal_cost_pred'] = y_pred_df.pred.map(
    lambda x: 1 if x > optimal_cost_cutoff_probability else 0
)

# Displaying the confusion matrix for cost-based predictions
get_cm(y_pred_df.actual, y_pred_df.optimal_cost_pred)

# %%
# %%

# Generating a classification report for logistic regression with cost-based analysis
classification_report_2 = metrics.classification_report(y_pred_df.actual,
                                                        y_pred_df.optimal_cost_pred)
print(classification_report_2)

# %%
# %%

# Building a decision tree classifier to predict the probability of CHD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# %%
# %%

# Performing hyperparameter tuning for optimal criterion and maximum depth
parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': range(2, 10)}]
clf_tree = DecisionTreeClassifier()
clf = GridSearchCV(clf_tree, parameters, cv=10, scoring='roc_auc')
clf.fit(X_train[significant_vars], y_train)

# %%
# %%

# Displaying the best score obtained from GridSearchCV
clf.best_score_

# %%
# %%

# Displaying the best hyperparameters obtained from GridSearchCV
clf.best_params_

# %%
# %%

# Building a decision tree classifier with the best parameters
clf_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5
)
clf_tree.fit(X_train[significant_vars], y_train)
y_pred_df['tree_predict'] = clf_tree.predict(X_test[significant_vars])

# %%
# %%

# Comparing the performance of logistic regression and decision tree models
roc_auc_score_clf_tree = metrics.roc_auc_score(y_pred_df.actual, y_pred_df.tree_predict)
roc_auc_score_youden_index = metrics.roc_auc_score(y_pred_df.actual, y_pred_df.optimal_pred)
roc_auc_score_cost = metrics.roc_auc_score(y_pred_df.actual, y_pred_df.optimal_cost_pred)
roc_auc_score_youden_index, roc_auc_score_cost, roc_auc_score_clf_tree

# %% [markdown]
#  # Logistic regression using Youden's Index provides the best performance among the three models.



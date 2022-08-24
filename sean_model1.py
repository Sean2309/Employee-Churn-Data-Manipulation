# General Standard Imports
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Dataset Preparation Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from imblearn.under_sampling import RandomUnderSampler

# Model Imports
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Model Eval Imports
from sklearn import metrics

#-----------------
# Input Variables
#-----------------
cat_cols = ["Pclass", "Sex", "Sibsp", "Cabin"]
X = ["Pclass", "Sex", "Age", "SibSp", "Cabin" ]
y = "Fare"


#-----------------
# Reading DF 
# Splitting train test
#-----------------
df = pd.read_csv("D://Code\datasets//titanic//train.csv", encoding="utf-8")
df = df.dropna()
df1 = pd.DataFrame()
for i in cat_cols:
    temp_df = pd.get_dummies(df[i], drop_first=True).add_prefix(f"{str(i)}_")
    df = df.drop(i, axis=1)
    df = pd.concat([df, temp_df], axis=1)

train_x, test_x, train_y, test_y = train_test_split(df1, df, test_size=0.33, random_state=42)

#------------
# Model Init
#------------
"""
Support Vector Machine
"""
# svc = SVC(kernel="linear")
# svc.fit(train_x_vector, train_y)

"""
Decision Tree
"""
def decision_tree(
    x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame, 
    test_y: pd.Series
):
    labels = y.unique()
    dec_tree = DecisionTreeClassifier(max_depth=3)
    dec_tree.fit(x, y)
    plt.figure(figsize=(30,10), facecolor="k")
    a = tree.plot_tree(
        dec_tree,
        feature_names=x.columns,
        class_names=labels,
        rounded=True,
        filled=True,
        fontsize=14
    )
    plt.show()


def logistic_reg(
    x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame, 
    test_y: pd.Series
):
    log_reg = LogisticRegression()
    log_reg.fit(x, y)
    print("=========LOGISTIC REGRESSION============")
    y_pred = log_reg.predict(test_x)
    df_pred = pd.DataFrame({
        "Actual": test_y.squeeze(),
        "Predicted": y_pred.squeeze()
    })
    print("\n")
    print(df_pred)
    print("=======================================")


def linear_reg(
    x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame, 
    test_y: pd.Series
):     
    x_name = x.columns[0]
    y_name = y.name
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    intercept = round(lin_reg.intercept_, 3)
    
    coeff = round(lin_reg.coef_[0], 3)
    print("=========LINEAR REGRESSION============\n")
    print("Intercept val: " + str(intercept) + "\n")

    if coeff < 0:
        print(f"Gradient Val is: {coeff} => This shows that there is a negative correlation between {y_name} and {x_name}")
    else:
        print(f"Gradient Val is: {coeff} => This shows that there is a positive correlation between {y_name} and {x_name}")

    y_pred = lin_reg.predict(test_x)
    df_pred = pd.DataFrame({
        "Actual": test_y.squeeze(),
        "Predicted": y_pred.squeeze()
    })
    print("\n")
    print(df_pred)
    print("=======================================")






#-------------------
# Calling Functions
#-------------------
# linear_reg(
#     x=train_x,
#     y=train_y, 
#     test_x=test_x,
#     test_y=test_y
# )

# logistic_reg(
#     x=train_x,
#     y=train_y, 
#     test_x=test_x,
#     test_y=test_y
# )

# decision_tree(
#     x=train_x,
#     y=train_y, 
#     test_x=test_x,
#     test_y=test_y
# )
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data
df = pd.read_csv("cybersecurity_attacks.csv")
print(df.info())

# Drop unnecessary data attributes (columns)
columns_to_drop = [
    "Timestamp",
    "Payload Data",
    "User Information",
    "Device Information",
    "Geo-location Data",
    "Proxy Information",
    "Firewall Logs",
    "IDS/IPS Alerts",
    "Log Source",
]

df.drop(columns=columns_to_drop, axis=1, inplace=True)

def has_missing_values(df):
    return df.isnull().values.any()

def missing_values_summary(df):
    summary = df.isnull().sum().sort_values(ascending=False)
    print(f"Shape of the dataframe is: {df.shape}\n")
    return summary.to_string()

print(f"Does the dataframe have any missing values? - {has_missing_values(df)}\n")
print(f"Missing values summary:\n{missing_values_summary(df)}")

def replace_missing_values(df, strategy):
    if strategy == "fill_with_constant":
        constant_to_fill = "Not Detected"
        df["Malware Indicators"].fillna(constant_to_fill, inplace=True)

        constant_to_fill = "No Alets"
        df["Alerts/Warnings"].fillna(constant_to_fill, inplace=True)
    elif strategy == "mean_imputation":
        # You can use this method for numerical features replacement
        df.fillna(df.mean(), inplace=True)

replace_missing_values(df, "fill_with_constant")
print(f"Missing values after replacement: {missing_values_summary(df)}")

# Categorical and numerical features
categorical_features = [
    "Source IP Address",
    "Destination IP Address",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Packet Type",
    "Traffic Type",
    "Alerts/Warnings",
    "Attack Type",
    "Attack Signature",
    "Action Taken",
    "Severity Level",
    "Network Segment",
]

numerical_features = ["Packet Length", "Anomaly Scores"]

# Data preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)
numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical_transformer", categorical_transformer, categorical_features),
        ("numerical_transformer", numerical_transformer, numerical_features),
    ]
)

# Split dataset
X = df.drop(columns=["Malware Indicators"])
y = df["Malware Indicators"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)



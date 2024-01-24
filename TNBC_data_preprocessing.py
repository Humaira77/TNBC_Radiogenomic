#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# File import
features_900 = pd.read_csv('/Users/humairanoor/Documents/Breast_Cancer/Radiomics_Features/Radiomics_features_900_smallsubset.csv', index_col=0)
risk_binary = pd.read_csv('/Users/humairanoor/Documents/Breast_Cancer/Eric_Signature/eric_total_risk_score.csv', index_col=0)

# Normalization of gene expression matrix

scaler = StandardScaler() 
X_normalized_900 = pd.DataFrame(scaler.fit_transform(features_900))

# Multicollinearity removal
correlation_matrix = X_normalized_900.corr()

# Set the correlation threshold
threshold = 0.65

# Find pairs of highly correlated features
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j] > threshold):
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# Remove highly correlated features
df_filtered = X_normalized_900.drop(columns=correlated_features)


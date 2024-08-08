#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.stats import kendalltau

X=df_filtered #0.65 threshold
y = risk_binary[['binary']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Number of top-k features to select
k = 20

# Calculate Kendall's rank coefficient for each feature in X_train
kendall_coeffs = {}
for feature in X_train.columns:
    tau, _ = kendalltau(X_train[feature], y_train)
    kendall_coeffs[feature] = abs(tau)

# Sort the features by Kendall's tau coefficient in descending order
selected_features = sorted(kendall_coeffs, key=kendall_coeffs.get, reverse=True)[:k]

X_train_subset = X_train[selected_features]
X_test_subset = X_test[selected_features]


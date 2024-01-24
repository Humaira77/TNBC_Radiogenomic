#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

# Import validation cohort

X_test_new = pd.read_csv('/Users/humairanoor/Documents/Breast_Cancer/Radiomics_Features/TNBC_194-63_subcohort_radiomics.csv', index_col=0)

# Filter dataframe for the selected features
selected_columns = [  470,
 476,
 322,
 41,
 405,
 328,
 10,
 766,
 785,
 488,
 239,
 489,
 100,
 3,
 109,
 523,
 414,
 487,
 544,
 153]
X_selected = X_test_new.iloc[:, selected_columns]

# Normalization
scaler = StandardScaler() 
X_test_new_norm = pd.DataFrame(scaler.fit_transform(X_selected))

# Prediction using the trained model

y_pred_new = classifier.predict(X_test_new_norm)
y_pred_new=pd.DataFrame(y_pred_new)
print(y_pred_new)
y_pred_new.to_csv('y_pred_decisiontree_kendall.csv')


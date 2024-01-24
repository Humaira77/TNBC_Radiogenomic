#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import numpy as np

# Define the number of repetitions
num_repetitions = 100

# Initialize lists to collect results
cv_scores_list = []
test_accuracy_list = []
test_precision_list = []
test_f1_list = []
test_recall_list = []
roc_auc_test_list = []
roc_auc_train_list = []
precision_train_list = []
f1_train_list = []
recall_train_list = []

# Repeat the process for the specified number of repetitions
for _ in range(num_repetitions):
    X_train, X_test, y_train_s, y_test = train_test_split(X_train_subset, y_train, test_size=0.3)
    
    classifier = DecisionTreeClassifier(ccp_alpha= 0.0, class_weight= None, criterion= 'entropy', max_depth= 30, max_features= None, max_leaf_nodes= None, min_impurity_decrease= 0.0, min_samples_leaf= 1, min_samples_split= 10, min_weight_fraction_leaf=0.0,  splitter='best')

    skfold = LeaveOneOut()
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train_s, cv=skfold)
    
    # Collect the cross-validation scores
    cv_scores_list.extend(cv_scores)
    
    classifier.fit(X_train, y_train_s)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)

    # Evaluate the model's accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_list.append(test_accuracy)
    
    test_precision = precision_score(y_test, y_pred)
    test_precision_list.append(test_precision)
    
    test_f1 = f1_score(y_test, y_pred)
    test_f1_list.append(test_f1)
    
    test_recall = recall_score(y_test, y_pred)
    test_recall_list.append(test_recall)
    
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]

    # Calculate the ROC AUC score for the test set
    roc_auc_test = roc_auc_score(y_test, y_pred_prob)
    roc_auc_test_list.append(roc_auc_test)

    y_pred_prob_train = classifier.predict_proba(X_train)[:, 1]

    # Calculate the ROC AUC score for the training set
    roc_auc_train = roc_auc_score(y_train_s, y_pred_prob_train)
    roc_auc_train_list.append(roc_auc_train)

    precision_train = precision_score(y_train_s, y_pred_train)
    precision_train_list.append(precision_train)

    f1_train = f1_score(y_train_s, y_pred_train)
    f1_train_list.append(f1_train)

    recall_train = recall_score(y_train_s, y_pred_train)
    recall_train_list.append(recall_train)

# Calculate the mean and standard deviation of the collected scores
mean_cv_score = np.mean(cv_scores_list)
mean_test_accuracy = np.mean(test_accuracy_list)
mean_test_precision = np.mean(test_precision_list)
mean_test_f1 = np.mean(test_f1_list)
mean_test_recall = np.mean(test_recall_list)
mean_roc_auc_test = np.mean(roc_auc_test_list)
mean_roc_auc_train = np.mean(roc_auc_train_list)
mean_precision_train = np.mean(precision_train_list)
mean_f1_train = np.mean(f1_train_list)
mean_recall_train = np.mean(recall_train_list)

# Print the aggregated results
print("Mean CV Score:", mean_cv_score)
print("Mean Test Accuracy:", mean_test_accuracy)
print("Mean Test Precision:", mean_test_precision)
print("Mean Test F1:", mean_test_f1)
print("Mean Test Recall:", mean_test_recall)
print("Mean ROC AUC Score Test:", mean_roc_auc_test)
print("Mean ROC AUC Score Train:", mean_roc_auc_train)
print("Mean Precision Train:", mean_precision_train)
print("Mean F1 Train:", mean_f1_train)
print("Mean Recall Train:", mean_recall_train)


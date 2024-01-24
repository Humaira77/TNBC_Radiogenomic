#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Make predictions on the test set
y_pred = classifier.predict(X_test)


    # Evaluate the model's accuracy
test_accuracy = accuracy_score(y_test, y_pred)

    
test_precision = precision_score(y_test, y_pred)

    
test_recall = recall_score(y_test, y_pred)
    
    
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

    # Calculate the ROC AUC score for the test set
roc_auc_test = roc_auc_score(y_test, y_pred_prob)
    
y_pred_prob_train = classifier.predict_proba(X_train_subset)[:, 1]

    # Calculate the ROC AUC score for the training set
roc_auc_train = roc_auc_score(y_train, y_pred_prob_train)
    

precision_train = precision_score(y_train, y_pred_train)
    
f1_train = f1_score(y_train, y_pred_train)
    

recall_train = recall_score(y_train, y_pred_train)


#----------------------------------#

# coding: utf-8, author: Nicolas Bourez
# date: 2024-05-09

#----------------------------------#

#----------------------------------#

# Importing the libraries

import pandas as pd
import numpy as np
import csv
import time
import cupy as cp

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

#----------------------------------#

# utils functions

def preprocess_standardization(X_train):

    numerical_features = X_train.select_dtypes(exclude=['category']).columns

    X_train_standardized = StandardScaler().fit_transform(X_train[numerical_features])

    return pd.DataFrame(data=X_train_standardized, index=X_train.index)

def balanced_accuracy(y_true,y_pred):
    """
    Balanced Classification Rate
    """
    # number of classes
    n_classes = len(set(y_true))
    # initialize the BCR
    bcr = 0
    # for each class
    for i in range(n_classes):
        # get the indices of the class i
        idx = y_true == i
        # compute the accuracy of the class i
        acc = accuracy_score(y_true[idx], y_pred[idx])
        # update the BCR
        bcr += acc
    # return the BCR
    return bcr / n_classes


def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)

#----------------------------------#

# data loading

data = pd.read_pickle('data/A5_2024_xtrain.gz')
target = pd.read_pickle('data/A5_2024_ytrain.gz')
to_predict = pd.read_pickle('data/A5_2024_xtest.gz')

# data preprocessing

# convert True/False to 1/0 for the 5002th column and 5003th column and make it a categorical variable
data.iloc[:, 5001] = pd.Categorical(data.iloc[:, 5001].astype(int))
data.iloc[:, 5002] = pd.Categorical(data.iloc[:, 5002].astype(int))
data.iloc[:, 5004] = pd.Categorical(data.iloc[:, 5004].astype(int))

to_predict.iloc[:, 5001] = pd.Categorical(to_predict.iloc[:, 5001].astype(int))
to_predict.iloc[:, 5002] = pd.Categorical(to_predict.iloc[:, 5002].astype(int))
to_predict.iloc[:, 5004] = pd.Categorical(to_predict.iloc[:, 5004].astype(int))

# Convert the target to a a classification problem knowing that the target has tree differents categorical values : "ER+/HER2-", "ER-/HER2-", "HER2+"
target = target.map({'ER+/HER2-': 0, 'ER-/HER2-': 1, 'HER2+': 2})

#----------------------------------#

# Feature selection for each model

#----------------------------------#

start_time = time.time()

# Global model

data_cp = data.copy()
to_predict_cp = to_predict.copy()

# Standardization
data_cp = preprocess_standardization(data_cp)
to_predict_cp = preprocess_standardization(to_predict_cp)

model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='multi:softprob', verbosity=0, eval_metric= "merror",random_state=42,device = 'cuda')

data_cp_gpu = cp.array(data_cp)
target_gpu = cp.array(target)
                      
model.fit(data_cp_gpu, target_gpu)

scores = model.feature_importances_
sorted_score = np.sort(scores)

# find the threshold where the score are higher than 0
threshold = sorted_score[sorted_score > 0]
thresh = threshold[0]

# select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(data_cp)
select_X_test = selection.transform(to_predict_cp)

data_selected_logit = pd.DataFrame(select_X_train)
x_test_selected_logit = pd.DataFrame(select_X_test)

elapsed_time = time.time() - start_time
print(f"\rTime elapsed after preprocessing global model : {elapsed_time:.2f} seconds", end="", flush=True)

# ----------------------------------#

# Binary model for class 1

target_ERminus = target.copy().map({0: 0, 1: 1, 2: 0})
weight_1 = target_ERminus.value_counts()[0] / target_ERminus.value_counts()[1]

data_cp1 = data.copy()
to_predict_cp1 = to_predict.copy()

# Standardization
data_cp1 = preprocess_standardization(data_cp1)
to_predict_cp1 = preprocess_standardization(to_predict_cp1)

model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='binary:logistic', verbosity=0, eval_metric='auc',scale_pos_weight=weight_1,random_state=42,device = 'cuda')

data_cp1_gpu = cp.array(data_cp1)
target_ERminus_gpu = cp.array(target_ERminus)
                              
model.fit(data_cp1_gpu, target_ERminus_gpu)

scores = model.feature_importances_
sorted_score = np.sort(scores)

# find the threshold where the score are higher than 0
threshold = sorted_score[sorted_score > 0]
thresh = threshold[0]

# select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(data_cp)
select_X_test = selection.transform(to_predict_cp)

data_selected_1 = pd.DataFrame(select_X_train)
x_test_selected_1 = pd.DataFrame(select_X_test)

elapsed_time = time.time() - start_time
print(f"\rTime elapsed after preprocessing binary model 1: {elapsed_time:.2f} seconds", end="", flush=True)

# ----------------------------------#

# Binary model for class 2

target_HER2 = target.copy().map({0: 0, 1: 0, 2: 1})
weight_2 = target_HER2.value_counts()[0] / target_HER2.value_counts()[1]

data_cp2 = data.copy()
to_predict_cp2 = to_predict.copy()

# Standardization
data_cp2 = preprocess_standardization(data_cp2)
to_predict_cp2 = preprocess_standardization(to_predict_cp2)

model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='binary:logistic', verbosity=0, eval_metric='auc',scale_pos_weight=weight_1,random_state=42,device = 'cuda')

data_cp2_gpu = cp.array(data_cp2)  
target_HER2_gpu = cp.array(target_HER2)

model.fit(data_cp2_gpu, target_HER2_gpu)

scores = model.feature_importances_
sorted_score = np.sort(scores)

# find the threshold where the score are higher than 0
threshold = sorted_score[sorted_score > 0]
thresh = threshold[0]

# select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(data_cp)
select_X_test = selection.transform(to_predict_cp)

data_selected_2 = pd.DataFrame(select_X_train)
x_test_selected_2 = pd.DataFrame(select_X_test)

elapsed_time = time.time() - start_time
print(f"\rTime elapsed after preprocessing binary model 2: {elapsed_time:.2f} seconds", end="", flush=True)

#----------------------------------#

# Model training and prediction

#----------------------------------#

data_lr = data_selected_logit.values
data_svm1 = data_selected_1.values
data_svm2 = data_selected_2.values

# my three models
model = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='multinomial',max_iter=2000)
model_svm1 = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='ovr',max_iter=2000)
model_svm2 = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='ovr',max_iter=2000)

# my three target variables
y_ERminus = target_ERminus.values
y_HER2 = target_HER2.values
y = target.values

#fitting
model.fit(data_lr, y)
model_svm1.fit(data_svm1, y_ERminus)
model_svm2.fit(data_svm2, y_HER2)

#predicting
y_pred = model.predict(x_test_selected_logit.values)
y_prob = model.predict_proba(x_test_selected_logit.values)
y_prob1 = model_svm1.predict_proba(x_test_selected_1.values)
y_prob2 = model_svm2.predict_proba(x_test_selected_2.values)

y_new_prob = np.zeros((len(y_pred),3))

for i in range(len(y_prob)):
    y_new_prob[i][0] = y_prob[i][0]*0.95
    y_new_prob[i][1] = y_prob1[i][1] if y_prob1[i][1] > y_prob[i][1] else y_prob[i][1]
    y_new_prob[i][2] = y_prob2[i][1]*1.05 if y_prob2[i][1] > y_prob[i][2] else y_prob[i][2]

y_new_pred = np.zeros(len(y_pred))  

for i in range(len(y_new_prob)):

    maximum = max(y_new_prob[i])

    if maximum == y_new_prob[i][0]:
        y_new_pred[i] = 0
    elif maximum == y_new_prob[i][1]:
        y_new_pred[i] = 1
    else :
        y_new_pred[i] = 2

# put y_prob, y_pred and y_test in the same dataframe
df = pd.DataFrame({'y_pred': y_pred, 'y_new_pred': y_new_pred ,'y_nprob_class0': y_new_prob[:, 0], 'y_nprob_class1': y_new_prob[:, 1], 'y_nprob_class2': y_new_prob[:, 2]})

# change the values of the target variable to the original values
y_pred = pd.Series(y_new_pred).map({0: 'ER+/HER2-', 1: 'ER-/HER2-', 2: 'HER2+'})

# name the y_pred as 'label'
y_pred.name = 'label'

# save the predictions
y_pred.to_csv('Predictions.csv', quoting=csv.QUOTE_NONNUMERIC, index=True)

elapsed_time = time.time() - start_time
print(f"\rTime elapsed after prediction : {elapsed_time:.2f} seconds", end="", flush=True)

#----------------------------------#

# Number of each class in the prediction

#----------------------------------#

print(f"\nNumber of each class in the prediction :")
print(df['y_new_pred'].value_counts())

#----------------------------------#

# Code for computing the expected balanced accuracy

#----------------------------------#

total_avg_bcr = 0

# -- Cross-validation
for i in range(5):
    kf = KFold(n_splits=10, shuffle=True)

    avg_acc, avg_bal_acc, avg_new_bcr = 0, 0, 0
    acc_bcr = []
    fold = 0

    data_copy = data.copy()

    for train_index, test_index in kf.split(data_copy.values):

        # -- Learning
        X_train, X_test = data_copy.iloc[train_index], data_copy.iloc[test_index]

        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        y_train_bin_1 = target_ERminus.iloc[train_index]
        y_train_bin_2 = target_HER2.iloc[train_index]

        X_train = preprocess_standardization(X_train)
        X_test = preprocess_standardization(X_test)

        model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='multi:softprob', verbosity=0, eval_metric= "merror",random_state=42,device='cuda')
        model_bin_1 = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='binary:logistic', verbosity=0, eval_metric='auc',scale_pos_weight=weight_1,random_state=42, device='cuda')
        model_bin_2 = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.03,subsample=0.75,objective='binary:logistic', verbosity=0, eval_metric='auc',scale_pos_weight=weight_2,random_state=42, device='cuda')

        X_train_cp = cp.array(X_train)
        X_test_cp = cp.array(X_test)
        y_train_cp = cp.array(y_train)
        y_train_bin_1_cp = cp.array(y_train_bin_1)
        y_train_bin_2_cp = cp.array(y_train_bin_2)

        # scores
        model.fit(X_train_cp, y_train_cp)
        model_bin_1.fit(X_train_cp, y_train_bin_1_cp)
        model_bin_2.fit(X_train_cp, y_train_bin_2_cp)

        scores = model.feature_importances_
        sorted_score = np.sort(scores)

        scores_bin_1 = model_bin_1.feature_importances_
        sorted_score_bin_1 = np.sort(scores_bin_1)

        scores_bin_2 = model_bin_2.feature_importances_
        sorted_score_bin_2 = np.sort(scores_bin_2)

        # find the threshold where the score are higher than 0
        threshold = sorted_score[sorted_score > 0]
        thresh = threshold[0]

        threshold_bin_1 = sorted_score_bin_1[sorted_score_bin_1 > 0]
        thresh_bin_1 = threshold_bin_1[0]

        threshold_bin_2 = sorted_score_bin_2[sorted_score_bin_2 > 0]
        thresh_bin_2 = threshold_bin_2[0]

        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        select_X_test = selection.transform(X_test)

        selection_bin_1 = SelectFromModel(model_bin_1, threshold=thresh_bin_1, prefit=True)
        select_X_train_bin_1 = selection_bin_1.transform(X_train)
        select_X_test_bin_1 = selection_bin_1.transform(X_test)

        selection_bin_2 = SelectFromModel(model_bin_2, threshold=thresh_bin_2, prefit=True)
        select_X_train_bin_2 = selection_bin_2.transform(X_train)
        select_X_test_bin_2 = selection_bin_2.transform(X_test)

        data_selected = pd.DataFrame(select_X_train)
        x_test_selected = pd.DataFrame(select_X_test)

        data_selected_bin_1 = pd.DataFrame(select_X_train_bin_1)
        x_test_selected_bin_1 = pd.DataFrame(select_X_test_bin_1)

        data_selected_bin_2 = pd.DataFrame(select_X_train_bin_2)
        x_test_selected_bin_2 = pd.DataFrame(select_X_test_bin_2)

        clf = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='multinomial',max_iter=2000)
        clf_bin_1 = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='ovr',max_iter=2000)
        clf_bin_2 = LogisticRegression(C=0.007, penalty='l2',class_weight='balanced',multi_class='ovr',max_iter=2000)

        clf.fit(data_selected, y_train)
        clf_bin_1.fit(data_selected_bin_1, y_train_bin_1)
        clf_bin_2.fit(data_selected_bin_2, y_train_bin_2)

        # -- Prediction
        y_pred = clf.predict(x_test_selected)

        y_prob = clf.predict_proba(x_test_selected)
        y_prob_bin_1 = clf_bin_1.predict_proba(x_test_selected_bin_1)
        y_prob_bin_2 = clf_bin_2.predict_proba(x_test_selected_bin_2)

        y_new_prob = np.zeros((len(y_pred),3))

        for i in range(len(y_prob)):
            y_new_prob[i][0] = y_prob[i][0]*0.9
            y_new_prob[i][1] = y_prob_bin_1[i][1] if y_prob_bin_1[i][1] > y_prob[i][1] else y_prob[i][1]
            y_new_prob[i][2] = y_prob_bin_2[i][1]*1. if y_prob_bin_2[i][1] > y_prob[i][2] else y_prob[i][2]

        y_new_pred = np.zeros(len(y_pred))  

        for i in range(len(y_new_prob)):
            maximum = max(y_new_prob[i])

            if maximum == y_new_prob[i][0]:
                y_new_pred[i] = 0
            elif maximum == y_new_prob[i][1]:
                y_new_pred[i] = 1
            else :
                y_new_pred[i] = 2
            
        # put y_prob, y_pred and y_test in the same dataframe
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'y_new_pred': y_new_pred ,'y_nprob_class0': y_prob[:, 0], 'y_nprob_class1': y_prob[:, 1], 'y_nprob_class2': y_prob[:, 2]})

        acc_score = accuracy_score(y_test, y_pred)
        bacc_score = balanced_accuracy(y_test, y_pred)
        new_bcr = balanced_accuracy(y_test, y_new_pred)

        avg_acc += acc_score
        avg_bal_acc += bacc_score
        avg_new_bcr += new_bcr
        acc_bcr.append(new_bcr)

        fold += 1

    print(f'accuracy: {avg_acc/10}')
    print(f'standard_bcr: {avg_bal_acc/10}')
    print(f'new_bcr: {avg_new_bcr/10}')
    print("Standard Deviation of Balanced Accuracy : ", np.std(acc_bcr))
    print('------------------------------------------------------------------------')

    total_avg_bcr += avg_new_bcr/10

print(f'Total Average Balanced Classification Rate : {total_avg_bcr/5}')
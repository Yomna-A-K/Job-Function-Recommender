import pandas as pd
project_path = 'C:\\Users\\Dell\\PycharmProjects\\JobFunctionRecommender\\'
'''test = pd.read_csv(project_path + 'dataset\\jobs_data_test.csv')
import pickle
with open(project_path + 'saved_models\\job_titles_binarizer.pkl', 'rb') as file:
    mlb_title_list_features = pickle.load(file)

original_columns = test.shape[1]
transformedTest = test.join(pd.DataFrame(columns=mlb_title_list_features.classes_).add_prefix('JT_'))
dummies_start_index = original_columns
dummies_end_index = transformedTest.shape[1]

import time
from model_scripts.job_title_feature_extraction import get_title_features
start_time = time.time()
job_titles_column_index = test.columns.get_loc("title")
for i in range(transformedTest.shape[0]):
    title_features = []
    row_features_dict = get_title_features(transformedTest.iloc[i,job_titles_column_index])
    for feature in row_features_dict:
        title_features.append(str(feature))
    row_values = mlb_title_list_features.transform([title_features])
    transformedTest.iloc[i, dummies_start_index:dummies_end_index] = row_values[0]

X_test = transformedTest.iloc[:,dummies_start_index:dummies_end_index]
print("------------------------------ %s minutes for title pre-processing-----------------------" % str((time.time() - start_time) / 60.0))

with open(project_path + 'saved_models\\job_functions_binarizer.pkl', 'rb') as file:
    mlb_functions_list_features = pickle.load(file)
original_columns = transformedTest.shape[1]
transformedTest = transformedTest.join(pd.DataFrame(columns=mlb_functions_list_features.classes_).add_prefix('F_'))
dummies_start_index = original_columns
dummies_end_index = transformedTest.shape[1]

import time
start_time = time.time()
job_functions_column_index = transformedTest.columns.get_loc("jobFunction")
for i in range(transformedTest.shape[0]):
    row_functions = transformedTest.iloc[i,job_functions_column_index].translate(str.maketrans({"'":None," ":None,"[":None,"]":None})).split(',')
    row_values = mlb_functions_list_features.transform([row_functions])
    transformedTest.iloc[i, dummies_start_index:dummies_end_index] = row_values[0]
y_test = transformedTest.iloc[:,dummies_start_index:dummies_end_index]
print("------------------------------ %s minutes for job functions pre-processing-----------------------" % str((time.time() - start_time) / 60.0))

transformedTest.to_csv(project_path + 'dataset\\jobs_data_test_processed.csv')

print("preprocessing successful")'''
import pickle
with open(project_path + 'saved_models\\TrainedLogisticClassifier.pkl', 'rb') as file:
    clf_log = pickle.load(file)

transformedTest = pd.read_csv(project_path + 'dataset\\jobs_data_test_processed.csv')
X_test = transformedTest.iloc[:,5:1864]
y_test = transformedTest.iloc[:,1864:1902]
import time
start_time = time.time()
Log_score = clf_log.score(X_test,y_test)
print(' Logistic Regression:', Log_score)
print("------------------------------ %s minutes for recommendation-----------------------" % str((time.time() - start_time) / 60.0))

start_time = time.time()
with open(project_path + 'saved_models\\TrainedRandomForestClassifier.pkl', 'rb') as file:
    clf_rfc = pickle.load(file)
RFC_score = clf_rfc.score(X_test,y_test)
print(' Random Forest max depth 100 random state 0:', RFC_score)
print("------------------------------ %s minutes for recommendation-----------------------" % str((time.time() - start_time) / 60.0))
start_time = time.time()
with open(project_path + 'saved_models\\TrainedKNNClassifier.pkl', 'rb') as file:
    clf_rfc = pickle.load(file)
KNN_Score = clf_rfc.score(X_test,y_test)
print(' KNN:', KNN_Score)
print("------------------------------ %s minutes for recommendation-----------------------" % str((time.time() - start_time) / 60.0))

import seaborn as sns
import matplotlib.pyplot as plt
x=["Random Forest", "Logistic Regression", "KNN"]
y = [RFC_score* 100,Log_score * 100,KNN_Score * 100]
plt.figure(figsize=(8,5))
ax = sns.barplot(x, y)
plt.title("Test Set Accuracy")
plt.ylabel('classification accuracy', fontsize=12)
plt.xlabel('Classification Model', fontsize=12)
plt.show()
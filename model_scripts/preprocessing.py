import pandas as pd
import numpy as np
#import os
#os.chdir("D:/JobFunctionRecommender/dataset")
project_path = 'C:\\Users\\Dell\\PycharmProjects\\JobFunctionRecommender\\'
df = pd.read_csv(project_path + 'dataset\\jobs_data_train.csv')
df = df.drop(df.columns[0], axis=1)

from sklearn.preprocessing import MultiLabelBinarizer
from model_scripts.job_title_feature_extraction import get_title_features

job_titles_list = df['title']
title_features = []
for i in range(job_titles_list.shape[0]):
    row_features = get_title_features(job_titles_list[i])
    for feature in row_features:
        title_features.append(str(feature))

mlb_title_list = MultiLabelBinarizer()
mlb_title_list.fit([title_features])
print("Number of unique title features is : " + str(len(mlb_title_list.classes_)))

original_columns = df.shape[1]
resDF = df.join(pd.DataFrame(columns=mlb_title_list.classes_).add_prefix('JT_'))
dummies_start_index = original_columns
dummies_end_index = resDF.shape[1]

import time
start_time = time.time()
job_titles_column_index = df.columns.get_loc("title")
for i in range(job_titles_list.shape[0]):
    title_features = []
    row_features_dict = get_title_features(resDF.iloc[i,job_titles_column_index])
    for feature in row_features_dict:
        title_features.append(str(feature))
    row_values = mlb_title_list.transform([title_features])
    resDF.iloc[i, dummies_start_index:dummies_end_index] = row_values[0]
    print(i)

print("------------------------------ %s minutes for title pre-processing-----------------------" % str((time.time() - start_time) / 60.0))
functions_list = resDF['jobFunction']
functions_list = np.array(functions_list).reshape(functions_list.shape[0],1)
all_functions = []
for i in range(functions_list.shape[0]):
    row_functions = functions_list[i,0].translate(str.maketrans({"'":None," ":None,"[":None,"]":None})).split(',')
    for j in range(len(row_functions)):
        all_functions.append(row_functions[j])

mlb_job_functions = MultiLabelBinarizer()
mlb_job_functions.fit([all_functions])
print("Number of unique job functions is : " + str(len(mlb_job_functions.classes_)))

original_columns = resDF.shape[1]
resDF = resDF.join(pd.DataFrame(columns=mlb_job_functions.classes_).add_prefix('F_'))
dummies_start_index = original_columns
dummies_end_index = resDF.shape[1]

import time
start_time = time.time()
job_functions_column_index = resDF.columns.get_loc("jobFunction")
for i in range(functions_list.shape[0]):
    row_functions = resDF.iloc[i,job_functions_column_index].translate(str.maketrans({"'":None," ":None,"[":None,"]":None})).split(',')
    row_values = mlb_job_functions.transform([row_functions])
    resDF.iloc[i, dummies_start_index:dummies_end_index] = row_values[0]
    print(i)
print("------------------------------ %s minutes for job functions pre-processing-----------------------" % str((time.time() - start_time) / 60.0))

industry_list = resDF['industry']
industry_list = np.array(industry_list).reshape(industry_list.shape[0],1)
all_industries = []
for i in range(industry_list.shape[0]):
    row_industries = industry_list[i,0].translate(str.maketrans({"'":None," ":None,"[":None,"]":None})).split(',')
    for j in range(len(row_industries)):
        all_industries.append(row_industries[j])

mlb_job_industries = MultiLabelBinarizer()
mlb_job_industries.fit([all_industries])
print("Number of unique industries is : " + str(len(mlb_job_industries.classes_)))

original_columns = resDF.shape[1]
resDF = resDF.join(pd.DataFrame(columns=mlb_job_industries.classes_).add_prefix('I_'))
dummies_start_index = original_columns
dummies_end_index = resDF.shape[1]

import time
start_time = time.time()
job_industries_column_index = resDF.columns.get_loc("industry")
for i in range(industry_list.shape[0]):
    row_industries = resDF.iloc[i,job_industries_column_index].translate(str.maketrans({"'":None," ":None,"[":None,"]":None})).split(',')
    row_values = mlb_job_industries.transform([row_industries])
    resDF.iloc[i, dummies_start_index:dummies_end_index] = row_values[0]
    print(i)
print("------------------------------ %s minutes for industry pre-processing-----------------------" % str((time.time() - start_time) / 60.0))

import pickle
with open(project_path + 'saved_models\\job_titles_binarizer.pkl', 'wb') as f:
    pickle.dump(mlb_title_list, f)

with open(project_path + 'saved_models\\job_functions_binarizer.pkl', 'wb') as f:
    pickle.dump(mlb_job_functions, f)

with open(project_path + 'saved_models\\job_industries_binarizer.pkl', 'wb') as f:
    pickle.dump(mlb_job_industries, f)

print('Saving models successful')

resDF.to_csv(project_path + 'dataset\\jobs_data_processed.csv')




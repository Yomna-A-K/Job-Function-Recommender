import pandas as pd
import pickle
import numpy as np

project_path = 'C:\\Users\\Dell\\PycharmProjects\\JobFunctionRecommender\\'
df = pd.read_csv(project_path + 'dataset\\jobs_data_processed.csv')
print(df.shape)

with open(project_path + 'saved_models\\job_titles_binarizer.pkl', 'rb') as file:
    mlb_title_list_features = pickle.load(file)

#print(mlb_title_list_features.classes_[130:150])
title_feature_start_index = 4
number_of_job_title_features = len(mlb_title_list_features.classes_)
title_feature_end_index = title_feature_start_index + number_of_job_title_features
titles_features = df.iloc[:,title_feature_start_index:title_feature_end_index]
title_features_occurences = np.sum(titles_features)
print(str(title_features_occurences))

with open(project_path + 'saved_models\\job_functions_binarizer.pkl', 'rb') as file:
    mlb_functions_list_features = pickle.load(file)

job_functions_start_index = title_feature_end_index
number_of_job_functions_features = len(mlb_functions_list_features.classes_)
job_functions_end_index = job_functions_start_index + number_of_job_functions_features
job_functions = df.iloc[:,job_functions_start_index: job_functions_end_index]
job_functions_occurences = np.sum(job_functions)
print(str(job_functions_occurences))

#ignore industry features for now
all_training_data = df.iloc[:,0:job_functions_end_index]
print(all_training_data.shape)
#df = df.drop(columns={'industry','jobFunction'},axis=1)
#print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt
rowsums = all_training_data.iloc[:,job_functions_start_index:job_functions_end_index].sum(axis=1)
x=rowsums.value_counts()
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple job functions per title")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of functions', fontsize=12)
plt.show()

X_train, y_train, y_train_original = all_training_data.iloc[:,title_feature_start_index:title_feature_end_index].values, all_training_data.iloc[:,job_functions_start_index:job_functions_end_index].values,all_training_data['jobFunction'].values
##X_test, y_test, y_test_original = test.iloc[:,title_feature_start_index:title_feature_end_index].values, test.iloc[:,job_functions_start_index:job_functions_end_index].values,test['jobFunction'].values

print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

clf = OneVsRestClassifier(MultinomialNB())
clf.fit(X_train,y_train)
#y_train_predicted_one_hot = clf.predict(X_train)
print(' Naive Bayes Accuracy:', clf.score(X_train,y_train))
import time
start_time = time.time()
from sklearn.linear_model import LogisticRegression
clf_log =  OneVsRestClassifier(LogisticRegression(C = 5,solver='lbfgs', max_iter=1000))
clf_log.fit(X_train,y_train)
print("------------------------------ %s minutes for logistic regression fit-----------------------" % str((time.time() - start_time) / 60.0))
#y_train_predicted_one_hot = clf.predict(X_train)
print(' Logistic regression Accuracy:', clf_log.score(X_train,y_train))
print("------------------------------ %s minutes for logistic regression score-----------------------" % str((time.time() - start_time) / 60.0))

with open(project_path + 'saved_models\\TrainedLogisticClassifier.pkl', 'wb') as f:
    pickle.dump(clf_log, f)
'''import time
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
start_time = time.time()
classifier = ClassifierChain(MultinomialNB(alpha=5))
# Training logistic regression model on train data
classifier.fit(X_train, y_train)
print(' Naive Bayes chain Accuracy --gave almost same accuracy as one vs rest no change:', classifier.score(X_test,y_test))
print("------------------------------ %s minutes for Chain RF score-----------------------" % str((time.time() - start_time) / 60.0))
#with open(project_path + 'saved_models\\TrainedChainRFClassifier.pkl', 'wb') as f:
    #pickle.dump(classifier, f)'''
import time
from sklearn.ensemble import RandomForestClassifier
clf_rfc = OneVsRestClassifier(RandomForestClassifier(max_depth=100, random_state=0))
start_time = time.time()
clf_rfc.fit(X_train,y_train)
print("------------------------------ %s minutes for RF fitting-----------------------" % str((time.time() - start_time) / 60.0))
print(' RandomForestClassifier:', clf_rfc.score(X_train,y_train))
y_test_predicted_one_hot = clf_rfc.predict(X_train)
print(' RandomForestClassifier:', clf_rfc.score(X_train,y_train))
for i in range(20):
    title_features = np.array(X_train[i,:]).reshape(1,number_of_job_title_features)
    title = mlb_title_list_features.inverse_transform(title_features)
    predicted_job_functions = np.array(y_test_predicted_one_hot[i,:]).reshape(1,number_of_job_functions_features)
    job_functions = mlb_functions_list_features.inverse_transform(predicted_job_functions)
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        str(title),(y_train_original[i]),(str(job_functions))
    ))


with open(project_path + 'saved_models\\TrainedRandomForestClassifier.pkl', 'wb') as f:
    pickle.dump(clf_rfc, f)

from sklearn.neighbors import KNeighborsClassifier
import time
start_time = time.time()
neigh = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
neigh.fit(X_train,y_train)
print("------------------------------ %s minutes for KNN fitting-----------------------" % str((time.time() - start_time) / 60.0))
start_time = time.time()
print(' KNN:', neigh.score(X_train,y_train))
print("------------------------------ %s minutes for KNN score-----------------------" % str((time.time() - start_time) / 60.0))
with open(project_path + 'saved_models\\TrainedKNNClassifier.pkl', 'wb') as f:
    pickle.dump(neigh, f)







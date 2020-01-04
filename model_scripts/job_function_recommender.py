
def job_function_recommender(job_title):
    #job_title = input()
    #import time
    #start_time = time.time()
    project_path = 'C:\\Users\\Dell\\PycharmProjects\\JobFunctionRecommender\\'
    import pickle
    with open(project_path + 'saved_models\\job_titles_binarizer.pkl', 'rb') as file:
        mlb_title_list_features = pickle.load(file)

    from model_scripts.job_title_feature_extraction import get_title_features
    title_features = get_title_features(job_title)
    row_title_features = []
    for feature in title_features:
        row_title_features.append(str(feature))
    row_values = mlb_title_list_features.transform([row_title_features])

    # print("------------------------------ %s seconds for title pre-processing-----------------------" % str((time.time() - start_time)))
    with open(project_path + 'saved_models\\TrainedRandomForestClassifier.pkl', 'rb') as file:
        clf_rfc = pickle.load(file)
    import numpy as np
    x_test = np.array(row_values[0]).reshape(1, -1)
    job_functions = clf_rfc.predict(x_test)
    # print("------------------------------ %s seconds for title classfication-----------------------" % str((time.time() - start_time)))
    with open(project_path + 'saved_models\\job_functions_binarizer.pkl', 'rb') as file:
        mlb_functions_list_features = pickle.load(file)

    predicted_job_functions = np.array(job_functions).reshape(1, -1)
    job_functions = mlb_functions_list_features.inverse_transform(predicted_job_functions)
    job_functions_list = str(job_functions).replace("'", "").replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').split(',')
    #for i in job_functions_list:
        #print(i)
    return job_functions_list
    #print("------------------------------ %s seconds for inverse transform-----------------------" % str((time.time() - start_time)))





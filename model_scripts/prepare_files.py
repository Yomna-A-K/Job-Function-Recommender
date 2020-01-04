import pandas as pd
from sklearn.model_selection import train_test_split
project_path = 'C:\\Users\\Dell\\PycharmProjects\\JobFunctionRecommender\\'
df = pd.read_csv(project_path + 'dataset\\jobs_data.csv')
df = df.drop(df.columns[0], axis=1)

train, test = train_test_split(df, random_state=42, test_size=0.1, shuffle=True)
train.to_csv(project_path + 'dataset\jobs_data_train.csv')
test.to_csv(project_path + 'dataset\jobs_data_test.csv')
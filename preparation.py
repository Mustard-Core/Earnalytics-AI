from nltk import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from settings import *

encoder = preprocessing.LabelEncoder()
le = LabelEncoder()

# Load dataset
df = pd.read_csv(r"Salary_Data.csv")
df = df.dropna()

#Changing Gender to numeric values
encoder.fit(['Female','Male','Other'])
df['Gender'] = encoder.fit_transform(df['Gender'])

encoder.fit(["Bachelor's","Master's", "PhD","Bachelor's Degree","Master's Degree","High School"])
df["Education Level"] = encoder.fit_transform(df["Education Level"])


arr_job_title = df['Job Title']
arr_job_title = arr_job_title.to_numpy()

new_list =[]

job_titles = ""

for i in range(len(arr_job_title)):
    if(arr_job_title[i] not in new_list):
        job_titles += arr_job_title[i] + " , "
        
new_list = sorted(sent_tokenize(job_titles))

#label encoding
encoder.fit(new_list)
df['Job Title'] = encoder.fit_transform(df['Job Title'])

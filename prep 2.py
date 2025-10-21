from nltk import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from settings import *

encoder = preprocessing.LabelEncoder()
le = LabelEncoder()

def convert(arrValue):
    # Load dataset
    df = pd.read_csv(r"Salary_Data.csv")
    df = df.dropna()
    
    new_data = [arrValue]

    new_df = pd.DataFrame(new_data, columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"])
    new_data = pd.concat([df, new_df])


    new_df = pd.concat([df, new_df])

    #Changing Gender to numeric values
    le.fit(df['Gender'])
    new_df["Gender"] = le.fit_transform(new_df['Gender'])

    encoder.fit(["Bachelor's","Master's", "PhD","Bachelor's Degree","Master's Degree","High School"])
    new_df["Education Level"] = le.fit_transform(new_df['Education Level'])

    arr_job_title = new_df['Job Title']
    arr_job_title = arr_job_title.to_numpy()

    new_list =[]
    job_titles = ""

    for i in range(len(arr_job_title)):
        if(arr_job_title[i] not in new_list):
            job_titles += arr_job_title[i] + " , "
            
    new_list = sorted(sent_tokenize(job_titles))

    #label encoding
    encoder.fit(new_list)
    new_df["Job Title"] = le.fit_transform(new_df['Job Title'])

    data = new_df.iloc[len(new_df)-1]
    data = data.to_numpy()
    return data


result = convert([32,"Male","Bachelor's","Business Analyst",7,0])

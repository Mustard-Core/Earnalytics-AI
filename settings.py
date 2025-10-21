import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model,svm
from sklearn.linear_model import LogisticRegression,LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import Bunch
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



encoder = preprocessing.LabelEncoder()

pd.set_option('display.max_rows', None) # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', None) # Auto width to avoid wrapping
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf)


def Logistic_visualize(Classifier_LR, X, y):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    mesh_step_size = 0.02
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),np.arange(min_y, max_y, mesh_step_size))


    output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1,cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),
    1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),
    1.0)))
    plt.show()






def encoding(labels, columnName):
    encoder.fit(labels)
    return encoder.fit_transform(df[columnName])


def convert(arrValue):

    encoder = preprocessing.LabelEncoder()
    le = LabelEncoder()

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
    data = data.astype(int)
    return data

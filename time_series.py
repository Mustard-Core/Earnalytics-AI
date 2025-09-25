from preparation import *
from settings import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(input_file):
    input_data = np.loadtxt(input_file, delimiter=None)
    dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    output = pd.Series(input_data[:, index], index=dates)
    return output




input_file = "Salary_Data.csv"



timeseries = pd.read_csv(input_file)

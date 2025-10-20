import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
import settings


dataframe = pandas.read_csv("housing.csv")
dataset = dataframe.values

X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = MLPClassifier(hidden_layer_sizes=(12,8), activation='relu', max_iter=150, batch_size=10, verbose=False)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

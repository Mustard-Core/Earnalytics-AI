import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
import settings


dataframe = pandas.read_csv("housing.csv")
dataset = dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]

def baseline_model():
    model_regressor = Sequential()
    model_regressor.add(Dense(13, input_dim=13, kernel_initializer='normal',activation='relu'))
    model_regressor.add(Dense(1, kernel_initializer='normal'))
    model_regressor.compile(loss='mean_squared_error', optimizer='adam')
    return model_regressor


seed = 7
numpy.random.seed(seed)

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5,verbose=0)
kfold = KFold(n_splits=10)
baseline_result = cross_val_score(estimator,X,Y,cv=kfold)

print("Baseline: %.2f (%.2f) MSE" %
(baseline_result.mean(),baseline_result.std()))


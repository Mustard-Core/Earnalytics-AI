import joblib


mj = joblib.load('model')

print(mj.predict( [[27,    1,    0,   14,    2 ]]))

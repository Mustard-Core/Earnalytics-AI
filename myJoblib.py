import joblib
import linear_regression as lr

def joblib_create_model(model, model_name):
    joblib.dump(model, model_name)

def joblib_create_model(model_name):
    return joblib.load(model_name)





y_pred = lr.model.predict(lr.X_test)
print(y_pred)

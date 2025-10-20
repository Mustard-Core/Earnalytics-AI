import joblib

import linear_regression as lr



# Suppose your trained model variable is named 'model'
joblib.dump(lr.model, "model")


y_pred = lr.model.predict(lr.X_test)
print(y_pred)

from preparation import *
from settings import *
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,median_absolute_error, mean_squared_error, r2_score,explained_variance_score
import myJoblib as jl
#Mapping features and target
X = df[["Age","Gender","Education Level","Job Title","Years of Experience"]]
X = X.to_numpy()

y = df['Salary']

#Train test split
test_size = 55
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size/100, random_state = 42)

#Linear regression
model = LinearRegression()

model = model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(y_test_pred)

score = model.score(X_test,y_test)
print("prediction score ", score* 100)

print("Performance of Linear regressor:")
print("Mean absolute error =", round(mean_absolute_error(y_test,y_test_pred), 2))
print("Mean squared error =", round(mean_squared_error(y_test, y_test_pred),2))
print("Median absolute error =", round(median_absolute_error(y_test,y_test_pred), 2))
print("Explain variance score =", round(explained_variance_score(y_test,y_test_pred), 2))
print("R2 score =", round(r2_score(y_test, y_test_pred), 2))


# = = = = = = = = = =



# Save the model
jl.save_model(model, "model")

# Later, to load the model
loaded_model = jl.load_model("model")



# hyperparameter tuning
# ================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline to scale data before regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Define the parameter grid
param_grid = {
    'regressor__fit_intercept': [True, False],
    'regressor__positive': [True, False]
}

# Grid Search (as shown in the YouTube tutorial)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation R² Score:", round(grid_search.best_score_, 3))

# Evaluate tuned model on test data
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nPerformance of Tuned Linear Regression:")
print("Mean Absolute Error =", round(mean_absolute_error(y_test, y_pred_tuned), 2))
print("Mean Squared Error =", round(mean_squared_error(y_test, y_pred_tuned), 2))
print("R2 Score =", round(r2_score(y_test, y_pred_tuned), 2))
print("Explained Variance =", round(explained_variance_score(y_test, y_pred_tuned), 2))


# Optimized GridSearchCV for LinearRegression (Better tuning with same structure)
# = = = = = = = = = =

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Recreate the pipeline with LinearRegression
pipeline_optimized = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Define the full parameter grid for LinearRegression
param_grid_optimized = {
    'regressor__fit_intercept': [True, False],
    'regressor__positive': [True, False]
}

# Use better CV and scoring strategy
grid_search_optimized = GridSearchCV(
    estimator=pipeline_optimized,
    param_grid=param_grid_optimized,
    scoring='neg_mean_squared_error',  
    cv=10,
    n_jobs=1
)

grid_search_optimized.fit(X_train, y_train)

print("\nBest Parameters from Optimized Grid Search:", grid_search_optimized.best_params_)
print("Best Cross-Validation MSE Score:", round(grid_search_optimized.best_score_, 3))

# Evaluate optimized model on test data
best_model_optimized = grid_search_optimized.best_estimator_
y_pred_optimized = best_model_optimized.predict(X_test)

print("\nPerformance of Optimized Tuned Linear Regression:")
print("Mean Absolute Error =", round(mean_absolute_error(y_test, y_pred_optimized), 2))
print("Mean Squared Error =", round(mean_squared_error(y_test, y_pred_optimized), 2))
print("R2 Score =", round(r2_score(y_test, y_pred_optimized), 2))
print("Explained Variance =", round(explained_variance_score(y_test, y_pred_optimized), 2))

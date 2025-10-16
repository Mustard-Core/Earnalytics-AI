import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("Salary_Data.csv")
df = df.dropna()

X = df.drop(columns=['Salary'])
y = df['Salary']

categorical_features = ['Gender', 'Education Level', 'Job Title']
numerical_features = ['Age', 'Years of Experience']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded_cat = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
X_encoded_cat_df = pd.DataFrame(X_encoded_cat, columns=encoded_feature_names, index=X.index)

scaler = MinMaxScaler()
X_scaled_num = scaler.fit_transform(X[numerical_features])
X_scaled_num_df = pd.DataFrame(X_scaled_num, columns=numerical_features, index=X.index)

X_processed = pd.concat([X_encoded_cat_df, X_scaled_num_df], axis=1)

test_size_ratio = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=test_size_ratio, random_state=42
)

model = Ridge(random_state=42)

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
}

print("Performing Grid Search Cross-Validation to fine-tune the Ridge Regression model...")
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters Found: ", grid_search.best_params_)

preds_tuned = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds_tuned)
r2 = r2_score(y_test, preds_tuned)

print("\nFine-Tuned Linear (Ridge) Regression Model Results:")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")

example_prediction = best_model.predict(X_test.iloc[[0]])
print(f"\nExample: Actual Salary: ${y_test.iloc[0]:.2f}, Predicted Salary: ${example_prediction[0]:.2f}")
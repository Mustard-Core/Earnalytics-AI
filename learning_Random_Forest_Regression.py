from preparation import *
from settings import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error, mean_squared_error, r2_score,explained_variance_score

#Mapping features and target
X = df[["Age","Gender","Education Level","Job Title","Years of Experience"]]

X = X.to_numpy()
y = df['Salary']

#Train test split
test_size = 70
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size/100, random_state = 42)

#Logistic regression
model = LinearRegression()
model = RandomForestRegressor(n_estimators=100, random_state=42)

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

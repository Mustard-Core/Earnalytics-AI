from preparation import *
from settings import *
from sklearn.linear_model import LogisticRegression

#Mapping features and target
X = df[['Years of Experience',"Age"]]

X = X.to_numpy()
y = df['Salary']


#Train test split
test_size = 20
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size/100, random_state = 42)

#Logistic regression

model = LinearRegression()
model = model.fit(X_train, y_train)
preds = model.predict(X_test)

print(preds)

score = model.score(X_test,y_test)
print("prediction score ", score* 100)
Logistic_visualize(model, X_train, y_train)




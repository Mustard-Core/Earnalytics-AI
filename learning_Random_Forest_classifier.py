from preparation import *
from settings import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

#Mapping features and target
X = df[["Age","Gender","Education Level","Job Title","Years of Experience"]]

X = X.to_numpy()
y = df['Salary']

#Train test split
test_size = 70
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size/100, random_state = 42)


#Train Random Forest Classification
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Make predictions
predictions = model.predict(X_test)

#Model Evaluation
#Accuracy
score = accuracy_score(y_test, predictions)
print("Score ", score* 100)


#precision
precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
print("Precision:", precision * 100)

# Recall
recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
print("Recall:", recall*100)

#F1-Score
f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
print("F1-score:", f1*100)








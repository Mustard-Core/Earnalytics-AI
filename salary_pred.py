import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from settings import *

encoder = preprocessing.LabelEncoder()
le = LabelEncoder()



# Load dataset
df = pd.read_csv(r"Salary_Data.csv")


df = df.dropna()

#Changing Gender to numeric values
encoder.fit(['Female','Male','Other'])
df['Gender'] = encoder.fit_transform(df['Gender'])



education_levels = ["Bachelor's","Master's", "PhD","Bachelor's Degree","Master's Degree","High School"]
encoder.fit(education_levels)
df['Education_Level'] = encoder.fit_transform(df['Education_Level'])



#changing the education to numeric values

df['Education_Level'] = df['Education_Level'].replace(["Bachelor's"],0)
df['Education_Level'] = df['Education_Level'].replace(["Master's"],1)
df['Education_Level'] = df['Education_Level'].replace(["PhD"],2)

#label encoding
job_titles = ['Account Manager','Accountant','Administrative Assistant','Business Analyst',
              'Business Development Manager','Business Intelligence Analyst','CEO','Chief Data Officer',
              'Chief Technology Officer','Content Marketing Manager','Copywriter','Creative Director',
              'Customer Service Manager','Customer Service Rep','Customer Service Representative',
              'Customer Success Manager','Customer Success Rep','Data Analyst','Data Entry Clerk',
              'Data Scientist','Digital Content Producer','Digital Marketing Manager','Director',
              'Director of Finance','Director of Human Resources','Director of Marketing','Director of Operations',
              'Director of Product Management','Director of Sales','Event Coordinator','Financial Advisor',
              'Financial Analyst','Financial Manager','Graphic Designer','Help Desk Analyst','HR Generalist',
              'HR Manager','Human Resources Director','IT Manager','IT Support','IT Support Specialist',
              'Junior Account Manager','Junior Accountant','Junior Business Analyst',
              'Junior Business Development Associate','Junior Copywriter','Junior Customer Support Specialist',
              'Junior Data Analyst','Junior Designer','Junior Developer','Junior Financial Analyst',
              'Junior HR Coordinator','Junior HR Generalist','Junior Marketing Analyst',
              'Junior Marketing Coordinator','Junior Marketing Manager','Junior Marketing Specialist',
              'Junior Operations Analyst','Junior Project Manager','Junior Recruiter','Junior Sales Representative',
              'Junior Software Developer','Junior Software Engineer','Junior Web Designer','Junior Web Developer',
              'Marketing Analyst','Marketing Coordinator','Marketing Manager','Marketing Specialist',
              'Network Engineer','Office Manager','Operations Analyst','Operations Director','Operations Manager',
              'Principal Engineer','Principal Scientist','Product Designer','Product Manager',
              'Product Marketing Manager','Project Engineer','Project Manager','Public Relations Manager',
              'Recruiter','Research Director','Research Scientist','Sales Associate','Sales Director',
              'Sales Executive','Sales Manager','Sales Operations Manager','Sales Representative',
              'Senior Account Manager','Senior Accountant','Senior Business Analyst',
              'Senior Business Development Manager','Senior Consultant','Senior Data Scientist',
              'Senior Engineer','Senior Financial Analyst','Senior Graphic Designer','Senior HR Generalist',
              'Senior HR Manager','Senior Human Resources Manager','Senior IT Support Specialist','Senior Manager',
              'Senior Marketing Analyst','Senior Marketing Coordinator','Senior Marketing Manager',
              'Senior Operations Manager','Senior Product Designer','Senior Product Manager',
              'Senior Product Marketing Manager','Senior Project Coordinator','Senior Project Manager',
              'Senior Research Scientist','Senior Researcher','Senior Sales Manager','Senior Sales Representative',
              'Senior Scientist','Senior Software Developer','Senior Software Engineer','Senior Training Specialist',
              'Social Media Manager','Social Media Specialist','Software Developer','Software Engineer',
              'Software Manager','Software Project Manager','Strategy Consultant','Supply Chain Analyst',
              'Supply Chain Manager','Technical Recruiter','Technical Support Specialist','Technical Writer',
              'Training Specialist','UX Designer','UX Researcher','VP of Finance','VP of Operations','Web Developer']

#encoding
encoder.fit(job_titles)
df['Job_Title'] = encoder.fit_transform(df['Job_Title'])


data = df.to_numpy()

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(data)

#Mapping features and target
X = data_scaled_minmax[:, :-1]  
y = data[:, -1]

#Train test split

test_size = 30
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size/100, random_state = 42)



#Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model = model.fit(X_train, y_train)
preds = model.predict(X_test)

print(preds)

score = model.score(X_test,y_test)

print("prediction score ", score* 100)

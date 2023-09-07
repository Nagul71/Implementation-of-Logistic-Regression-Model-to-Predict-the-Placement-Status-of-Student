# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array. 
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:


Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: K.NAGUL

RegisterNumber:  212222230089
```python 
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

## 1.Placement Data
![Screenshot 2023-09-07 081707](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/0dd174c9-5101-44ef-bc5e-c6d171f3cec2)


## 2.Salary Data

![Screenshot 2023-09-07 081714](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/4b767ce9-af77-4404-9b45-340bc563a8d7)

## 3. Checking the null function()
![Screenshot 2023-09-07 081720](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/5ac0e16a-c838-4869-82ef-dd81379a5ebc)


## 4.Data Duplicates
![Screenshot 2023-09-07 081727](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/aa4bfd35-4ffd-4865-860d-c01a2ad284e8)


## 5.Print Data
![Screenshot 2023-09-07 081742](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/c0c9beef-d486-4fe4-b95c-dd0e7c5dffbd)

## 6.Data Status
![Screenshot 2023-09-07 081749](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/e3cf5223-73ca-442e-bbbc-51806b337c8f)




## 7.y_prediction array
![Screenshot 2023-09-07 081757](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/2c8f1a5e-f02b-453e-99b1-52bd5614d9a1)

## 8.Accuracy value
![Screenshot 2023-09-07 081802](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/9c48a365-f927-478b-a892-e13dd9f75a23)

## 9.Confusion matrix
![Screenshot 2023-09-07 081806](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/b6012349-05ea-4b0b-a71a-fe04e095fdcb)




## 10.Classification Report
![Screenshot 2023-09-07 081811](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/e207d47e-b8f0-491c-8052-08f7aad243e0)


## 11.Prediction of LR
![Screenshot 2023-09-07 081816](https://github.com/Nagul71/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118661118/af74f0f3-2a3a-4790-8411-3e596f8ab9c9)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
STEP 1:
Import the required packages.

STEP 2:
Import the dataset to operate on.

STEP 3:
Split the dataset.

STEP 4:
Predict the required output.

STEP 5:
End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:Ragul M 
RegisterNumber:212221230080  
*/
```
~~~
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='latin-1')
data.head()
data.info()data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()                         #CountVectorizer is method to convert text into numerical data.The text is transformed to a sparse matrix
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~
## Output:
## Data Head:
![pic 1](https://github.com/ragulmani936/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/data%20head.png)
## Data info:
![pic 2](https://github.com/ragulmani936/Implementation-of-SVM-For-Spam-Mail-Detection/blob/main/data%20info.png)
## Data isnull:
![pic 3]()

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

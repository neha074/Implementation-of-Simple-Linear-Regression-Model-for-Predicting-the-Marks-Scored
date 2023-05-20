# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python.

2.Set variables for assigning data set values.

3.Import Linear Regression from the sklearn.

4.ssign the points for representing the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtain the LinearRegression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Neha.MA
RegisterNumber:  212220040100
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/content/student_scores.csv')

data.head()

data.tail()

x=data.iloc[:,:-1].values  
y=data.iloc[:,1].values

print(x)
print(y)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0 )

regressor=LinearRegression() 
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) 
print(y_pred)

print(y_test)

#for train values
plt.scatter(x_train,y_train) 
plt.plot(x_train,regressor.predict(x_train),color='black') 
plt.title("Hours Vs Score(Training set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

#for test values
y_pred=regressor.predict(x_test) 
plt.scatter(x_test,y_test) 
plt.plot(x_test,regressor.predict(x_test),color='black') 
plt.title("Hours Vs Score(Test set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(x, y)
mse = metrics.mean_squared_error(x, y)
rmse = np.sqrt(mse)  

print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

Output:
Head and Tail for the dataframe

![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/96f6863a-7363-4382-affb-edb927fd4204)

![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/7ad06fe2-68f4-4813-a0d7-739764ae7122)


Array value of X 


![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/bce88029-70b2-4492-83fa-0e1b747e109a)



Array Value of Y -


![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/7ce5027c-087a-437b-b37f-ce127f801da8)



Values of Y prediction and Y test


![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/939b79ae-38a2-47f5-9a04-19eb2c0a6c7d)


![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/31fd9618-4f36-43a6-8344-53ddf7298896)




Training Set Graph

![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/a130f19b-2be0-475a-bb64-951912cf4baf)



Test Set Graph

![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/813c0418-22e8-4b45-8f1a-75c53bc57265)




Values of MSE , MAE and RMSE

![image](https://github.com/neha074/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113016903/fea422f0-5a6f-4c91-ab81-b561ec3d5166)


Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# bhavana1995
task 1
Data Science And Business Analytics Task 1-Copy2
In [18]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
url='http://bit.ly/w-data'
s_data=pd.read_csv(url)
s_data.head(10)
Out[18]:
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25

Show more
In [20]:
s_data.plot(x='Hours', y='Scores', style = 'o') 
plt.title('Hours_Study v/s Percentage_Score') 
plt.xlabel('Hours_Study') 
plt.ylabel("Percentage_Score") 
plt.show()

In [21]:
X=s_data.iloc[:, :-1].values 
y=s_data.iloc[:, 1].values
In [22]:
X
Out[22]:
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
In [23]:
y
Out[23]:
array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
       24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)
In [24]:
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
In [27]:
from sklearn.linear_model import LinearRegression
Regressor1=LinearRegression()
Regressor1.fit(X,y)
Out[27]:
LinearRegression()
In [28]:
line = Regressor1.coef_*X+Regressor1.intercept_
plt. scatter(X,y)
plt.plot(X, line);
plt.show()

In [29]:
print(X_test)
y_pred=Regressor1.predict(X_test)
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
In [30]:
df = pd. DataFrame({'Actual':y_test, 'Predicted': y_pred})
df
Out[30]:
Actual	Predicted
0	20	17.147378
1	27	33.766244
2	69	74.824618
3	30	26.923182
4	62	60.160913
In [31]:
print("Training Score: ", Regressor1.score(X_train,y_train)) 
print("Testing Score: ", Regressor1.score (X_test,y_test))
Training Score:  0.9512837351709387
Testing Score:  0.9491748734859172
In [32]:
print("Training Score: ", Regressor1.score(X_train,y_train)) 
print("Testing Score: ", Regressor1.score (X_test,y_test))
Training Score:  0.9512837351709387
Testing Score:  0.9491748734859172
In [33]:
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1) 
own_pred = Regressor1.predict(test)
print("No of Hours = {}".format (hours))
print("Predicted Score ={}".format (own_pred[0]))
No of Hours = 9.25
Predicted Score =92.9098547701573
In [34]:
import numpy as np 
from sklearn import metrics

print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred)) 
print("Mean Squared Error:" ,metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error:" ,np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Explained Variance Score:" ,metrics.explained_variance_score(y_test,y_pred))
Mean Absolute Error: 4.071877793635605
Mean Squared Error: 20.138948129940175
Root Mean Squared Error: 4.487643939746131
Explained Variance Score: 0.9515224335188082

```PY
#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#Read the dataset
data=pd.read_csv("Churn_Modelling.csv")
data.head(5)
data.info()
# Finding Missing Values
data.isnull().sum()
data
#Check for Duplicates
data.duplicated()
#Normalize the dataset
scaler=MinMaxScaler()
data1=pd.DataFrame(scaler.fit_transform(data))
data1
#split the dataset into input and output
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x
y
#splitting the data for training & Testing
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.5,random_state=40)
#Detect Outliers
data.describe()
#Print the training data and testing data
print("x_train\n\n",x_train)
print("\nLenght of X_train ",len(x_train))
print("X_test\n",x_test)
print("\nLenght of X_test ",len(x_test))
```
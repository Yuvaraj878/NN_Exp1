<H3>ENTER YOUR NAME : YUVARAJ S</H3>
<H3>ENTER YOUR REGISTER NO : 212222240119</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 23/2/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
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
## OUTPUT:
### Checking Data :
![](./img/1.png)
### Missing Data :
![](./img/2.png)
### Duplicated Data :
![](./img/3.png)
### Normalization of Data :
![](./img/4.png)
### Outliers :
![](./img/5.png)
### Training and Testing Model :
![](./img/6.png)
![](./img/7.png)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



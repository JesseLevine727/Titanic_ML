from sklearn import datasets, linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")

#1.Data Pre-procesessing
#2.Splitting Data into Training and Test Set
#3.Applying Chosen Algo
#4.Test the accuracy


#1.a --> Divide data into categorical/Numerical

titanic_cat = titanic.select_dtypes(object)
titanic_num = titanic.select_dtypes(np.number)

titanic_test_cat = titanic_test.select_dtypes(object)
titanic_test_num = titanic_test.select_dtypes(np.number)

#1.b --> Removing un-important Columns and taking care of Nulls
titanic_cat.drop(['Name','Ticket'],axis=1,inplace=True)
titanic_num.drop(['PassengerId'],axis=1,inplace=True)

titanic_test_cat.drop(['Name','Ticket'],axis=1,inplace=True)
titanic_test_num.drop(['PassengerId'],axis=1,inplace=True)



titanic_cat.Cabin.fillna(titanic_cat.Cabin.value_counts().idxmax(),inplace=True)
titanic_cat.Embarked.fillna(titanic_cat.Embarked.value_counts().idxmax(),inplace=True)

titanic_test_cat.Cabin.fillna(titanic_test_cat.Cabin.value_counts().idxmax(),inplace=True)
titanic_test_cat.Embarked.fillna(titanic_test_cat.Embarked.value_counts().idxmax(),inplace=True)


titanic_num.Age.fillna(titanic_num.Age.mean(),inplace=True)
titanic_test_num.Age.fillna(titanic_test_num.Age.mean(),inplace=True)
titanic_test_num.Fare.fillna(titanic_test_num.Fare.mean(),inplace=True)

#1.c --> changing values to numbers
le = LabelEncoder()

titanic_cat = titanic_cat.apply(le.fit_transform)
titanic_test_cat = titanic_test_cat.apply(le.fit_transform)

#1.d --> combine data frames
titanic_final = pd.concat([titanic_cat,titanic_num],axis=1)
titanic_test_final = pd.concat([titanic_test_cat,titanic_test_num],axis=1)

#2. Partitioning Data

x_1 = titanic_final.drop(['Survived'],axis=1)
y_1 = titanic_final['Survived']

x_train = np.array(x_1[0:int(1*len(x_1))])
y_train = np.array(y_1[0:int(1*len(y_1))])

x_test = np.array(titanic_test_final[0:int(1*len(titanic_test_final)):])

#3. Applying Chosing algo
#I chose Random Forest

RF = RandomForestClassifier()

RF_fit = RF.fit(x_train,y_train)

RF_pred = RF_fit.predict(x_test)
print(RF_pred)

#4. Confidence
RF_pred_proba = RF_fit.predict_proba(x_test)
print(RF_pred_proba)



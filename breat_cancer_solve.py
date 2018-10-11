import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:].values
y = dataset.iloc[:,1].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=.2)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(y)
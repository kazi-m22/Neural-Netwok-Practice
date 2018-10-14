import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,6:13].values
y = dataset.iloc[:,13].values
# labelencoder_X_1=LabelEncoder() #create object
# X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

# labelencoder_X_2 = LabelEncoder()
# X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'tanh', input_dim = 7))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

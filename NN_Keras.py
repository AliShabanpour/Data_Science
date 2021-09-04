import pandas as pd


df = pd.read_csv("out.csv")

from sklearn.metrics import accuracy_score
target = df['Survived']
predictors = df.drop("Survived",axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.30,random_state=0)


## Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20,activation='relu',input_dim=8))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=500)
Y_pred_nn = model.predict(X_test)


rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")
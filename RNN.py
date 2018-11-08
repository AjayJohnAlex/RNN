# data preprocessing
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#importing the training set 
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#feature scaling 
#by either standardisation or normalisation(we would use normalisation)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a adata structure with 60 timestampss with 1 outputs
#with each 60 timstamps stops overstepping and it corresponds to 60 finanicial days
# so that means 3 months ;so we are predicting the next day's stock market after analysing 3 months data
#first entity X_train which willbe input of NN
#second would contains the output
X_train = []
y_train = []
for i in range(60,1258):#upperbound is the total no of rows(5 years )
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train , y_train = np.array(X_train), np.array(y_train)
#adding more dimensionality to the data 
# we have only 1 indicator
#reshaping to add a dimension in numpy array
#right now our Ds has 2 dimens. to make it to a 3d 
#coz RNN need input in a 3d tensor with batch_size, timestamps and input_dim
X_train  = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#building the RNN 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initilaising the RNN
regressor  = Sequential()
#layer1
#adding LSTM layer and Dropout regularisation
#dropout is done to remove over fitting
#we need to input these as argument in LSTM
# 1. the no of units(no of mem ubits ) 
#2. the return sequence (whether to return to lasst input)
# 3. the input shape with 2 dimens as 3rd dimens is already taken 
regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1],1) ))
#dropout rate is to be mentioned 
regressor.add(Dropout(0.2)) 
#layer2
regressor.add(LSTM(units = 50,return_sequences = True))
#dropout rate is to be mentioned 
regressor.add(Dropout(0.2))
#layer3
regressor.add(LSTM(units = 50,return_sequences = True))
#dropout rate is to be mentioned 
 
#layer4
regressor.add(LSTM(units = 50))
#dropout rate is to be mentioned 
regressor.add(Dropout(0.2))
#output layer 
regressor.add(Dense(units= 1))
#compiling the RNN
regressor.compile(optimizer='adam',loss= 'mean_squared_error')
#fit the RNN to training set 
regressor.fit(X_train,y_train,epochs=100,batch_size=32 )
# making th epredictions and doing visualisation
# we will real stock prize of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
actual_stock_price = dataset_test.iloc[:,1:2].values
#get the predicted stock prize of 2017 
# we need to concatenate the entire dataset
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
#getting the values of the inputs 
inputs = dataset_total[len(dataset_total) -  len(dataset_test) - 60 :].values
#reshaping in 1 column
inputs = inputs.reshape(-1,1)
#scale the predicting values 
inputs = sc.transform(inputs)
#again change into 3d shape to make the RNN predict the values 
X_test = []

for i in range(60,80):#upperbound is now to predict test set of 20 financial days 
    X_test.append(inputs[i-60:i,0])
    
X_test  = np.array(X_test)
#3d format
X_test  = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
# to get the original scale of values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#  Visualisation of the results
plt.plot(actual_stock_price,color= 'red',label = 'Real Google Stock Price')
plt.plot(predicted_stock_price,color= 'blue',label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
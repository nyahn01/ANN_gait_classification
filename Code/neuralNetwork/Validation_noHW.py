#We want to compare the result without using the data of height and weight
from tensorflow import keras
import pickle
import numpy as np

# Neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6, 100)),
    keras.layers.Dense(64,activation='elu'),
    keras.layers.Dense(32,activation='elu'),
    keras.layers.Dense(16,activation='elu'),
    keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Read in the data
f = open("input.pickle",'rb')
data_input = pickle.load(f)
f1 = open("target.pickle",'rb')
data_target = pickle.load(f1)
#Remove the information of height and weight
data_input = data_input[:,2:8,:]

#We use the 5-fold validation method to compare different models and tune the parameters
#The training set + The validation set has 90% of the dataset
#The remaining 10% part is the test set

#Cut the dataset into five folds
size1 = np.int(data_input.shape[0]*0.18)
size2 = np.int(data_input.shape[0]*0.36)
size3 = np.int(data_input.shape[0]*0.54)
size4 = np.int(data_input.shape[0]*0.72)
size5 = np.int(data_input.shape[0]*0.90)

# The first fold
train_input1 = data_input[:size4]
train_target1 = data_target[:size4]
valid_input1 = data_input[size4:size5]
valid_target1 = data_target[size4:size5]
#first train the network using the training set
model.fit(train_input1,train_target1,epochs=20)
#Use the validation set to fine tune the network parameters
loss1, accuracy1 = model.evaluate(valid_input1, valid_target1, verbose=2)


#The second fold
# The process of training and cross-validation
train_input2 = data_input[size1:size5]
train_target2 = data_target[size1:size5]
valid_input2 = data_input[:size1]
valid_target2 = data_target[:size1]
#first train the network using the training set
model.fit(train_input2,train_target2,epochs=20)
#Use the validation set to fine tune the network parameters
loss2, accuracy2 = model.evaluate(valid_input2, valid_target2, verbose=2)


#The third fold
# The process of training and cross-validation
train_input31 = data_input[:size1]
train_target31= data_target[:size1]
train_input32 = data_input[size2:size5]
train_target32= data_target[size2:size5]
train_input3 = np.concatenate([train_input31,train_input32],axis=0)
train_target3 = np.concatenate([train_target31,train_target32],axis=0)
valid_input3 = data_input[size1:size2]
valid_target3 = data_target[size1:size2]
#first train the network using the training set
model.fit(train_input3,train_target3,epochs=20)
#Use the validation set to fine tune the network parameters
loss3, accuracy3 = model.evaluate(valid_input3, valid_target3, verbose=2)


#The fourth fold
# The process of training and cross-validation
train_input41 = data_input[:size2]
train_target41= data_target[:size2]
train_input42 = data_input[size3:size5]
train_target42= data_target[size3:size5]
train_input4 = np.concatenate([train_input41,train_input42],axis=0)
train_target4 = np.concatenate([train_target41,train_target42],axis=0)
valid_input4 = data_input[size2:size3]
valid_target4 = data_target[size2:size3]
#first train the network using the training set
model.fit(train_input4,train_target4,epochs=20)
#Use the validation set to fine tune the network parameters
loss4, accuracy4 = model.evaluate(valid_input4, valid_target4, verbose=2)


#The fifth fold
# The process of training and cross-validation
train_input51 = data_input[:size3]
train_target51= data_target[:size3]
train_input52 = data_input[size4:size5]
train_target52= data_target[size4:size5]
train_input5 = np.concatenate([train_input51,train_input52],axis=0)
train_target5 = np.concatenate([train_target51,train_target52],axis=0)
valid_input5 = data_input[size3:size4]
valid_target5 = data_target[size3:size4]
#first train the network using the training set
model.fit(train_input5,train_target5,epochs=20)
#Use the validation set to fine tune the network parameters
loss5, accuracy5 = model.evaluate(valid_input5, valid_target5, verbose=2)

accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy5) / 5
print(accuracy)



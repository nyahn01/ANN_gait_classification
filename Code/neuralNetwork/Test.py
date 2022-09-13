from tensorflow import keras
import pickle
import numpy as np

# Neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8, 100)),
    keras.layers.Dense(64,activation='elu'),
    keras.layers.Dense(32,activation='elu'),
    keras.layers.Dense(16,activation='elu'),
    keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Read in the data
f = open("input.pickle",'rb')
data_input = pickle.load(f)
print(data_input.shape)
f1 = open("target.pickle",'rb')
data_target = pickle.load(f1)
print(data_target.shape)

mean_of_input = np.mean(data_input,axis=0)
max_of_input = np.max(data_input,axis=0)
min_of_input = np.min(data_input,axis=0)
data_input = (data_input - mean_of_input) / (max_of_input - min_of_input)

#After getting the network model and fine tuning the parameters
#The training set now is:  training set + validation set
train_size = np.int(data_input.shape[0]*0.9)
train_input1 = data_input[:train_size]
train_target1 = data_target[:train_size]
test_input = data_input[train_size:]
test_target = data_target[train_size:]

#first train the network using the training set
model.fit(train_input1,train_target1,epochs=15)

#Use the validation set to fine tune the network parameters
loss1, accuracy1 = model.evaluate(test_input, test_target, verbose=2)


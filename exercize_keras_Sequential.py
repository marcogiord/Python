# loading your own data tf and keras 

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle 

import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard
import time
####################################
###### LOADING data ################
#####################################
DATADIR = "D:/PetImages"

CATEGORIES = ["Dog", "Cat"]

NAME = "Cats-vs-dogs-CNN-10ep"

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#        plt.imshow(img_array, cmap='gray')  # graph it
#        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
    
print(img_array)

print(img_array.shape)


IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

############################################
###### creat etaining data  ################
############################################

full_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                full_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(full_data))

random.shuffle(full_data)

for sample in full_data[:10]:
    print(sample[1])
    

# creta ethe data     
X = []
y = []
training_data = full_data
#test_data = full_data[7001:8000]

for features,label in training_data[:]:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save and/ or loa ddata
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#We can always load it in to our current script, or a totally new one by doing:

#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)
#
#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)

#######################
##### Network with 2 cnn-layers-256, 1 Dense layer  
#https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/

#X = X/255.0
#
#model = Sequential()
#
#model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#
#model.add(Dense(64))
#model.add(Activation('relu'))
#
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#
#tensorboard = TensorBoard(log_dir="D:/Code/logs/{}".format(NAME))
#
#
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

#https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/?completed=/convolutional-neural-network-deep-learning-python-tensorflow-keras/


#model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
#model.fit(X, y,
#          batch_size=32,
#          epochs=10,
#          validation_split=0.3,
#          callbacks=[tensorboard])


# optimize 
#https://pythonprogramming.net/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/?completed=/tensorboard-analysis-deep-learning-python-tensorflow-keras/#
X = X/255.0
dense_layers = [0]
layer_sizes = [32]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            #NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            NAME = '64x3-CNN'
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])
            
model.save('64x3-CNN-10epochs.model')

TESTDIR = "D:/PetImagesTest/"
#test_data_array = []
##test_data_labels = []
#
#def create_test_data():
#    for category in CATEGORIES:  # do dogs and cats
#
#        path = os.path.join(TESTDIR,category)  # create path to dogs and cats
#        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
#
#        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
#            try:
#                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
#                test_data_array.append([img_array, class_num])  # add this to our training_data
#            except Exception as e:  # in the interest in keeping the output clean...
#                pass
#
#
#create_test_data()
#
##random.shuffle(test_data_array)
#
#X_test = []
#Y_test = []
#for feature,label in test_data_array[:]:
#    X_test.append(feature)
#    Y_test.append(label)
#    
#X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
#use the network to predict 
#https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/?completed=/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/
#def prepare(img_array):
#    IMG_SIZE = 100  # 50 in txt-based
#    #img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
#    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
#    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
#
##load the mmodel 
model = tf.keras.models.load_model("64x3-CNN.model")
prediction_list = []


for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(TESTDIR,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        Xt = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(Xt)
        prediction_class = CATEGORIES[int(round(prediction[0][0]))]
        print("predicitton is "+ str(prediction_class)+ " class is "+ CATEGORIES[class_num])    
        




counter = 0
num_rightp = 0
for img in X_test:
    print()
    prediction = model.predict([prepare(img)])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PRED
    print("predicitton is "+ str(prediction))    
    prediction_list.append(CATEGORIES[int(round(prediction[0][0]))])
    counter = counter + 1
    if (CATEGORIES[int(prediction[0][0])] == Y_test[counter]):
        num_rightp = num_rightp + 1
        
print("accuracy is "+ (num_rightp / X_test.size()))
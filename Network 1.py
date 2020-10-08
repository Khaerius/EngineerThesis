import os
import numpy as np
import scipy
import matplotlib
import cv2
import matplotlib.pyplot as plt
import keras
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NAME = 'Model1'

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

IMG_SIZE=224
keras.backend.set_image_data_format("channels_first")

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir,d))]
    data=[]
    image=[]
    for d in directories:
        label_dir = os.path.join(data_dir,d)
        file_names = [os.path.join(label_dir,f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            image = plt.imread(f)
            new_image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
            data.append([new_image,d])
    return data

training_data = load_data("Badania/Obrazy/artists10/")

random.shuffle(training_data)

Xrgb=[]
Xgr=[]
yrgb=[]
ygr=[]
dislabels=[]

for features,labels in training_data:
    if features.shape == (IMG_SIZE,IMG_SIZE,3):
        if labels not in dislabels:
            dislabels.append(labels)
        Xrgb.append(features)
        yrgb.append(dislabels.index(labels))
    else:
        Xgr.append(features)
        ygr.append(labels)
    
Xrgb = np.array(Xrgb).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
#Xgr = np.array(Xgr).reshape(-1, 1, IMG_SIZE, IMG_SIZE)

Xrgb = Xrgb.astype('float32')
Xrgb = Xrgb/255.0

yrgb = keras.utils.to_categorical(yrgb,10)

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=Xrgb.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['categorical_accuracy','accuracy'])

history = model.fit(Xrgb, yrgb, batch_size=32, 
                    epochs=10, callbacks=[tensorboard], validation_split=0.1, shuffle=True)

model.save('Model1')
                                                                




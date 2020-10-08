import os
import numpy as np
import scipy
import matplotlib
import cv2
import matplotlib.pyplot as plt
import keras
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
from keras.callbacks import TensorBoard

NAME = 'Model4'

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

IMG_SIZE=150
#keras.backend.set_image_data_format("channels_first")

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

model = keras.models.Sequential()
model.add(VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(Xrgb, yrgb, batch_size=32, epochs=15, callbacks=[tensorboard], validation_split=0.1, shuffle=True)

model.save('Model4')


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K

from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


PATH = os.getcwd()
data_path = 'C:\\Users\\User\\Desktop\\project\\dataset'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel= 3
num_epoch=3
 
num_classes = 2

labels_name = {'bike':0,'car':1}

img_data_list=[]
labels_list =[]
 
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print('Loaded the images of dataset- '+'{}\n'.format(dataset))
    label = labels_name[dataset]
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+dataset+'/'+img)
        #input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)
        labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /=255
print (img_data.shape)

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples),dtype='int64')
labels[0:1900]=0
labels[1900:]=1

names = ['bike','car']

#print the count of the number of samples for different classes
print(np.unique(labels,return_counts=True))

#Convert Class labels to on-hot encoding
Y = np_utils.to_categorical(labels,num_classes)

#shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

#split the dataset

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


#defining the model

input_shape = img_data[0].shape


model = Sequential()

model.add(Conv2D(32,3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=["accuracy"])

#Viewing model Configuration

model.summary()
model.get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


hist = model.fit(X_train,y_train, batch_size=16,nb_epoch=num_epoch,verbose=1,validation_data=(X_test,y_test))

from keras import callbacks

filename = 'model_train_new.csv'
csv_log=callbacks.CSVLogger(filename,separator=',',append=False)

early_stopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',verbose=1,save_best_only=True, mode='min')

callbacks_list=[csv_log,early_stopping,checkpoint]

hist = model.fit(X_train,y_train,batch_size=16,nb_epoch=num_epoch,verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)


model.save('./data.yml')





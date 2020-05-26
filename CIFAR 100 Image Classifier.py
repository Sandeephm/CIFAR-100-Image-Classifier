# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:58:26 2020

@author: Computer
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

NO_EPOCH=450
BATCH_SIZE=100

output_save_path='E:\Artificial Itelligence with Python'

(x_train, y_train),(x_test1, y_test1) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
x_test, x_val, y_test, y_val = train_test_split(x_test1,y_test1, test_size=0.5)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

# Normalize the train test and validation data
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_val = tf.keras.utils.normalize(x_val,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(100, activation="softmax"))

checkpoint = tf.keras.callbacks.ModelCheckpoint('E:/Artificial Itelligence with Python/best_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
print(model.summary())

# Train neural network
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=NO_EPOCH,batch_size=BATCH_SIZE,validation_data=(x_val,y_val),verbose=1, callbacks=[checkpoint])
model.save("E:/Artificial Itelligence with Python/model.h5")

model.evaluate(x_test,  y_test, verbose=2)

history_dict = history.history

with open(output_save_path+'\\'+'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history_dict, file_pi)

plt.figure()
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Acccuracy')
plt.legend()
plt.title('Acccuracy')
plt.savefig(output_save_path+'\\'+'Accuracy.png')
plt.close()


plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.savefig(output_save_path+'\\'+'Loss.png')
plt.close()
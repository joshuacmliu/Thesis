# cnn model with batch normalization for mnist
#%%
import pdb
import os
from re import X
from numpy.lib.function_base import average 
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import mean
from numpy import std
import matplotlib as plt
from matplotlib import pyplot
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
os.chdir("C:/Users/Joshua")
#from keras.layers import BatchNormalization
#%%

#%%
# load train and test dataset
def load_dataset():
    # load dataset
    x = np.load("X3.npy")
    y = np.load("y3.npy")
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY
#%%

#%%
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm
#%%

#%%
# define cnn model
def define_model(x, y):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(x.shape[1:])))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#%%

#%%
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model(dataX, dataY)
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        #weight classes to reduce imbalance of data
        trainY_int = np.argmax(trainY, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(trainY_int), trainY_int)
        class_weights_dict = {i:v for i,v in enumerate(class_weights)}
        # fit model
        history = model.fit(trainX, trainY, epochs=20, class_weight = class_weights_dict, batch_size=16, validation_data=(testX, testY), verbose = True)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories
#%%

#%%
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    loss, val_loss, accuracy, val_accuracy = list(), list(), list(), list()
    for i in range(len(histories)):
        loss.append(histories[i].history["loss"])
        val_loss.append(histories[i].history['val_loss'])
        accuracy.append(histories[i].history['accuracy'])
        val_accuracy.append(histories[i].history['val_accuracy'])
    
    loss = np.array(loss)
    loss = np.average(loss, axis=0)
    val_loss = np.array(val_loss)
    val_loss = np.average(val_loss, axis=0)
    accuracy = np.array(accuracy)
    accuracy = np.average(accuracy, axis = 0)
    val_accuracy = np.array(val_accuracy)
    val_accuracy = np.average(val_accuracy, axis = 0)
    #plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(loss, color='blue', label='train')
    pyplot.plot(val_loss, color='orange', label ="test")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    #plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(accuracy, color='blue', label='train')
    pyplot.plot(val_accuracy, color='orange', label='test')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy")
    pyplot.legend()    
    pyplot.tight_layout()
    pyplot.show()
    #     pyplot.subplot(2, 1, 1)
    #     pyplot.title('Cross Entropy Loss')
    #     pyplot.plot(histories[i].history["loss"], color='blue', label='train')
    #     pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
    #     # plot accuracy
    #     pyplot.subplot(2, 1, 2)
    #     pyplot.title('Classification Accuracy')
    #     pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
    #     pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    # pyplot.show()
#%%

#%%
# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    # pyplot.boxplot(scores)
    # pyplot.show()
#%%

#%%
# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()
#%%
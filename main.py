# Sequeira, Flyn
# 1001-778-678
# 2020_04_20
# Assignment-04-03

import pytest
import numpy as np
from cnn import CNN
import os
from tensorflow.keras.datasets import cifar10


def test_train_and_evaluate():
  # Initializing and adding layers
  print("*********** PLEASE WAIT FOR DATA TO LOAD ***********")
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
  train_images, test_images = train_images / 255.0, test_images / 255.0
  new_cnn = CNN()
  new_cnn.add_input_layer(shape=(32, 32, 3),name="input" )
  new_cnn.append_conv2d_layer(32, strides=3, activation="relu",name="conv2d_1")
  new_cnn.append_maxpooling2d_layer(pool_size=2, name="maxpool_1")
  new_cnn.append_conv2d_layer(64,strides=3, activation="relu", name="conv2d_2")
  new_cnn.append_maxpooling2d_layer(pool_size=2, name="maxpool_2")
  new_cnn.append_conv2d_layer(64,strides=3, activation="relu", name="conv2d_3")
  new_cnn.append_flatten_layer(name="flatten")
  new_cnn.append_dense_layer(64,activation="relu",name="dense_1")
  new_cnn.append_dense_layer(10,activation="softmax",name="dense_2")
  # Setting Compiler values
  new_cnn.set_loss_function(loss="SparseCategoricalCrossentropy")
  new_cnn.set_optimizer(optimizer="SGD")
  new_cnn.set_metric('accuracy')
  # Entering Num Epoch
  batch_size = 1000
  num_epoch = 10
  history = new_cnn.train(train_images, train_labels, batch_size=batch_size, num_epochs=num_epoch)
  assert len(history) == len(train_images)/batch_size*num_epoch
  evaluate = new_cnn.evaluate(train_images, train_labels)
  assert evaluate[1]<=1
  assert evaluate[0]<=3
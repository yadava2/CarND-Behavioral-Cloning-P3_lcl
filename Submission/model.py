import json
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import tensorflow as tf
import data_process

tf.python.control_flow_ops = tf

epochs = 8
learning_rate = 1e-4


# model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper

model = Sequential()

# Normalization Layer
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# Convolution layer 1 with RELU activation
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# Generators for training and validation
train_gen = data_process.generate_batch()
valid_gen = data_process.generate_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=20000,
                              nb_epoch=epochs,
                              validation_data=valid_gen,
                              nb_val_samples=6400,
                              verbose=1)

# finally save our model and weights

model.save('model.h5')



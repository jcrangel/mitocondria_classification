# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import layers
from keras import regularizers


class VggnetVer1:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3),input_shape=inputShape))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Conv2D(64, (3, 3), 
		activation='relu',kernel_regularizer=regularizers.l2(0.01)))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Conv2D(128, (3, 3), 
		activation='relu',kernel_regularizer=regularizers.l2(0.01)))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Conv2D(128, (3, 3), 
		activation='relu',kernel_regularizer=regularizers.l2(0.01)))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Flatten())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(512, activation='relu',
		kernel_regularizer=regularizers.l2(0.01)))
		model.add(layers.Dropout(0.7))
		model.add(layers.Dense(classes, activation=finalAct))

		# return the constructed network architecture
		return model

#Classic Vgg16net
class Vgg16net:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(Conv2D(64, (3, 3), input_shape=inputShape,
		padding='same', activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		model.add(Dense(4096, activation='relu'))
		model.add(Dense(classes, activation=finalAct))

		return model
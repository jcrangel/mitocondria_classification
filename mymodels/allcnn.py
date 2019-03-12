# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import regularizers
from keras.regularizers import l1

## Full cnn Buda et at 2018
class AllCNN:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax",weights=None):
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

		model.add(Conv2D(96, (3, 3),input_shape=inputShape, strides=(1,1),padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(96, (3, 3),strides=(1,1),padding='same'))
		model.add(Activation('relu'))		
		model.add(Conv2D(96, (3, 3),strides=(2,2),padding='same'))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Conv2D(192, (3, 3),strides=(1,1),padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(192, (3, 3),strides=(1,1),padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(192, (3, 3),strides=(2,2),padding='same'))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))		
		model.add(Conv2D(192, (3, 3),strides=(1,1),padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(192, (1, 1),strides=(1,1),padding='valid'))
		model.add(Activation('relu'))
		model.add(Conv2D(classes, (1, 1),strides=(1,1),padding='valid'))
		model.add(Activation('relu'))
		model.add(GlobalAveragePooling2D())
		model.add(Activation(finalAct))

		return model

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

#model from https://towardsdatascience.com/building-a-blood-cell-classification-model-using-keras-and-tfjs-5f7601ace931
class TowardsDataScienceModel:
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

        model.add(Conv2D(64, (3,3), strides = (1, 1), activation = 'relu', 
        input_shape=inputShape))
        model.add(Conv2D(80, (3,3), strides = (1, 1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation=finalAct))
        return model
        #model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

#https://doi.org/10.1109/JBHI.2016.2526603
class GaoModel:
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


        model.add(
        layers.Conv2D(6, (7, 7), activation='tanh', input_shape=inputShape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (4, 4), activation='tanh'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='tanh'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(150, activation='tanh'))
            #   model.add(layers.Dropout(0.7))
        model.add(Dense(classes, activation=finalAct))
        return model

class Cells5:
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


        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape , name= 'conv1'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',name= 'conv2' ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu',name= 'conv3' ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu',name= 'conv4' ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        #Seems that keras need name for this layers or will try to load the weights on it
        model.add(layers.Dense(512, activation='relu',name = 'new_dense')) # need name 
        model.add(layers.Dropout(0.7))
        model.add(Dense(classes, activation=finalAct, name = 'new_dense2')) #

        if weights is not None:
            model.load_weights(weights, by_name=True)

        return model    
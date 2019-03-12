from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K 
import keras.layers as layers



class ResNet50:
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

        conv_base = applications.resnet50.ResNet50(weights = "imagenet", 
        include_top=False, input_shape = inputShape)

        #Froze layers except the top ones        
        for layer in conv_base.layers:
            layer.trainable = False
        
        #Adding custom Layers 
        #Adding custom Layers 
        # x = model.output
        # x = Flatten()(x)
        # x = Dense(1024, activation="relu")(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation="relu")(x)
        # predictions = Dense(classes, activation=finalAct)(x)
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation="relu"))
        model.add(layers.Dropout(0.7))
        model.add(layers.Dense(1024, activation="relu"))
        model.add(Dense(classes, activation=finalAct))


        return model

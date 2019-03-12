# USAGE
#  python train.py  --column Classification --model cell_Classification.model \
# --dataset cell_image/crop_images/

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from mymodels.smallervggnet import SmallerVGGNet_v2
from mymodels.smallervggnet import SmallerVGGNet
from mymodels.vggnetmod1 import Vgg16net
from mymodels.others import TowardsDataScienceModel
from mymodels.others import GaoModel
from keras.preprocessing.image import img_to_array
from keras import regularizers, optimizers
from keras.optimizers import Adam
from collections import Counter
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import numpy as np
import datetime
import argparse
import time 
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-c", "--column", required=True,
	help="column to classify in the csv file")    
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
DEBUG = False
if not DEBUG:
    EPOCHS = 30
else: 
    EPOCHS = 1

INIT_LR = 1e-3
BS = 128
IMAGE_DIMS = (128, 128, 3)
MODEL =  SmallerVGGNet
#PATHS CONFIG
IMAGES_PATH = args['dataset']
PLOTS_DIR = 'plot_and_log'
LOG_FILE = os.path.sep.join([PLOTS_DIR,'log.txt'])
now = datetime.datetime.now()
now_ =  now.strftime("%Y-%m-%d_%H:%M") 
plot_name = 'plot' + now_   
PLOT_FILE = os.path.sep.join([PLOTS_DIR,plot_name])

log = '<---------------------------------------------------> \n'
log += 'Using Keras imgage  aug the same that chollet uses in the book \n'
log += 'And we split the data un just traning and validation no test since \n'
log += 'we have to few data \n'
log += 'Image dims : ' + str(IMAGE_DIMS) + '\n'
log += 'Train batch size : ' + str(BS) + '\n'
log += 'Model name : ' + str(MODEL.__name__) + '\n'
log += 'Plot file name : ' + PLOT_FILE + '\n'
log += 'EPOCHS : ' + str(EPOCHS) + '\n'

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
traindf=pd.read_csv("cell_image/cell_train.csv",dtype=str) # see preparingdata.ipynb
valdf=pd.read_csv("cell_image/cell_test.csv",dtype=str)
#testdf=pd.read_csv("cell_image/cell_test.csv",dtype=str)

# construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")

# Using one file for train and using validation split is not working
# The split leave out some classes: 
# Class 4 missing
#(Pdb) Counter(train_generator.classes) 
#Counter({1: 698, 2: 383, 7: 136, 0: 126, 8: 123, 5: 100, 3: 90, 6: 29}) 
# Class 1,3,5,6,7 missing
#Counter(valid_generator.classes)
#Counter({2: 315, 4: 98, 8: 27, 0: 8})  
train_datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator=train_datagen.flow_from_dataframe(
dataframe=traindf,
directory=IMAGES_PATH,
x_col="image_filename",
y_col=args['column'],
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))

val_datagen=ImageDataGenerator(rescale=1./255.)
valid_generator=val_datagen.flow_from_dataframe(
dataframe=valdf,
directory=IMAGES_PATH,
x_col="image_filename",
y_col=args['column'],
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))

# test_datagen=ImageDataGenerator(rescale=1./255.)

# test_generator=test_datagen.flow_from_dataframe(
# dataframe=testdf,
# directory=IMAGES_PATH,
# x_col="image_filename",
# y_col=args['column'], # None
# batch_size=30,
# seed=42,
# shuffle=False,
# class_mode="categorical", # None
# target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))



# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weighting = {class_id : max_val/num_images for class_id, num_images in counter.items()}   

finalAct = 'sigmoid'
model = MODEL.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(counter),
	finalAct=finalAct)
model.summary()
log += 'Final act : ' + finalAct + '\n'

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

log += 'Optimizer: ' + opt.__class__.__name__ +'\n'

# compile the model
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
loss="categorical_crossentropy",metrics=["accuracy"])

# train the network
print("[INFO] training network...")
if not DEBUG:
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
else: 
    STEP_SIZE_TRAIN=1
    STEP_SIZE_VALID=1

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

start = time.time()
H = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    class_weight = class_weighting ,
                    callbacks=[tensorboard]
)
end = time.time()
log += 'Training time: ' +str( (end - start)/60) + ' min \n'

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")

plt.savefig(PLOT_FILE)

#write the log to file
with open(LOG_FILE,'a+') as f:
    f.write(log)
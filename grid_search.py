#  Grid search
#  Just save the records to be examined on tensorboard

#  USAGE
#  python train.py  --column Classification --dataset cell_image/aug_image

#TODO modify cell_class copyright etc


# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
import mymodels
from keras.callbacks import ModelCheckpoint
from mymodels.resnet import ResNet
from keras import backend as K


from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Nadam
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import time 
import os

def train_model(desc='GridSearch',learning_rate=0.000016083,decay=0.0008061,beta_1=0.9,beta_2 = 0.999):
    print ('Parameter: learning rate:' + str(learning_rate) +
    ', ' + 'decay : ' + str(decay) )
    #Compiling
    print("[INFO] compiling model...")
    counter = Counter(train_generator.classes)                          
    finalAct = 'softmax'
    model = MODEL.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(counter),finalAct=finalAct)         

    INIT_LR = learning_rate
    DECAY = decay
    opt = Adam(lr=INIT_LR, beta_1=beta_1, beta_2=beta_2, decay=DECAY, amsgrad=False)
    model.compile(opt,loss="categorical_crossentropy",metrics=["accuracy"])


    #Tensorboard 
    tb_log_name  =desc+str(MODEL) +'beta_1:'+ str(beta_1) + '_' \
        + DESC.replace(' ','_') + 'beta_2:'+str(beta_2)
    tensorboard  = TensorBoard(log_dir="logs/{}".format( tb_log_name ))

    callbacks_list = [tensorboard]


    # train the network
    print("[INFO] training network...")
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    start = time.time()

    fit_args = dict(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=EPOCHS,
                        callbacks=callbacks_list)



    H = model.fit_generator(**fit_args)


    del model
    K.clear_session()
    





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-c", "--column", required=True,
	help="column to classify in the csv file")    

args = ap.parse_args()

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30 
BS = 128
IMAGE_DIMS = (128, 128, 3)
MODEL = mymodels.VGG19_frozen.VGG19
DESC = 'Resnet no weights' #Short description of the training 
BASE_PATH_IMG = args.dataset  #'cell_image/aug_images'
print("[INFO] loading images from " + BASE_PATH_IMG )
TRAIN_DIR = os.path.sep.join([BASE_PATH_IMG,args.column,'train'])
VAL_DIR = os.path.sep.join([BASE_PATH_IMG,args.column,'validation'])


train_datagen=ImageDataGenerator(rescale=1./255.,)

train_generator=train_datagen.flow_from_directory(
directory=TRAIN_DIR,
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))

val_datagen=ImageDataGenerator(rescale=1./255.)

valid_generator=val_datagen.flow_from_directory(
directory=VAL_DIR,
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))


#GRID search [1e-3, 1e-6]
# learning_rates = [10 ** (np.random.uniform(-3,-6) ) for i in range(0,10) ]
# decays =  [10 ** (np.random.uniform(-3,-6) ) for i in range(0,10) ]

# for lr in learning_rates:
#     for decay in decays:
#          train_model(lr,decay)

betas_1 = [0,0.9]
betas_2 = [0.99,0.999,0.9999]

for beta_1 in betas_1:
    for beta_2 in betas_2:
         train_model(desc='GridSearch3',beta_1=beta_1,beta_2=beta_2)





#Good findings:
# INIT_LR = (3.7796e-5 - 5.63e-6) / 2  = 0.000016083
#  DECAY = 0.0008061



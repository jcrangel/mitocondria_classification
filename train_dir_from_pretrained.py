# USAGE
#  python train.py  --column Classification --model cell_Classification.model \
# --dataset cell_image/crop_images/

#TODO modify cell_class copyright etc


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
import mymodels
from keras.callbacks import ModelCheckpoint


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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-c", "--column", required=True,
	help="column to classify in the csv file")    
ap.add_argument("-w", "--weights", action = 'store_true',
default=False,help="Use weights to balance the data training")  

args = ap.parse_args()

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
DEBUG = False
if not DEBUG:
    EPOCHS = 500 
else: 
    EPOCHS = 1

BS = 128
IMAGE_DIMS = (128, 128, 3)
MODEL =  mymodels.VGG19_frozen.VGG19
DESC = 'Random weights' 
#PATHS CONFIG
#IMAGES_PATH = args['dataset']
PLOTS_DIR = 'plot_and_log'
LOG_FILE = os.path.sep.join([PLOTS_DIR,'log.txt'])
now = datetime.datetime.now()
now_ =  now.strftime("%Y-%m-%d_%H:%M") 
plot_name = 'plot' + now_   
PLOT_FILE = os.path.sep.join([PLOTS_DIR,plot_name])

log = '<---------------------------------------------------> \n'
log += DESC + '\n'
log += 'Image dims : ' + str(IMAGE_DIMS) + '\n'
log += 'Train batch size : ' + str(BS) + '\n'
log += 'Model name : ' + str(MODEL.__name__) + '\n'
log += 'Plot file name : ' + PLOT_FILE + '\n'
log += 'EPOCHS : ' + str(EPOCHS) + '\n'

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")

BASE_PATH_IMG = 'cell_image/aug_images'
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

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
counter = Counter(train_generator.classes)                          
# max_val = float(max(counter.values()))       
# class_weighting = {class_id : max_val/num_images for class_id, num_images in counter.items()}   

finalAct = 'softmax'
#weights = 'pretrained_models/cells5_adam_noaug.h5'
model = MODEL.build(	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(counter),
	finalAct=finalAct)
# Resnet special build       
model.summary()
log += 'Final act : ' + finalAct + '\n'

INIT_LR = 1e-3   #((1e-3 - 1e-4) / 2) + 1e-3
DECAY = 1e-6
beta_2 = 0.999
# opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=beta_2, decay=DECAY, amsgrad=False)
opt = Nadam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#opt = SGD(lr=INIT_LR, momentum=0.9)
log += 'beta_2 = ' + str(beta_2) +'\n'
log += 'Optimizer: ' + opt.__class__.__name__ +'\n'
log += 'Learning rate: ' + str(INIT_LR) + '\n'
# compile the model
model.compile(opt,loss="categorical_crossentropy",metrics=["accuracy"])

tb_log_name  =str(MODEL) +'_lr:'+ str(INIT_LR) + '_' \
    + DESC.replace(' ','_') + 'Decay:'+str(DECAY) +'_'+ str(time.time())
tensorboard  = TensorBoard(log_dir="logs/{}".format( tb_log_name ))
# train the network
print("[INFO] training network...")
if not DEBUG:
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
else: 
    STEP_SIZE_TRAIN=1
    STEP_SIZE_VALID=1
# checkpoint
little_desc ='class_weights'
filepath=str(MODEL.__name__) +little_desc + \
    "_{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = os.path.sep.join(['checkpoints',filepath])
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
             verbose=1, save_best_only=True, mode='max')
callbacks_list = [tensorboard]

#Class weights 
class_weighting = None
if args.weights :
    print('[INFO] Calculating class weights')
    counter = Counter(valid_generator.classes)                          
    max_val = float(max(counter.values()))
    log += 'Class weights were used\n'       
    class_weighting = {class_id : max_val/num_images for class_id, num_images in counter.items()} 


start = time.time()
H = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=callbacks_list,
                    class_weight = class_weighting 
)
end = time.time()
log += 'Training time: ' +str( (end - start)/60) + ' min \n'

# save the model to disk
print("[INFO] serializing network...")
model.save(args.model)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.subplot(211)
plt.title("Training Loss and accuracy")
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.subplot(212)
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig(PLOT_FILE)

#write the log to file
with open(LOG_FILE,'a+') as f:
    f.write(log)

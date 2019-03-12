# USAGE
#  python train.py  --column Classification --model cell_Classification.model \
# --dataset cell_image/aug_image

#TODO modify cell_class copyright etc
#TODO modify lsuv.py


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
import mymodels
from keras.callbacks import ModelCheckpoint
from mymodels.resnet import ResNet
from utils.lsuv import *


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
ap.add_argument("--checkpoints", action = 'store_true',
default=False,help="Checkpoint the best model so far") 
ap.add_argument("--desc", default='ClassificationAug1',
 help="Description for the files in chekpoints") 
ap.add_argument("--weight-init", type=str,
                        help='The custom weight initializer ')   

args = ap.parse_args()

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30 
BS = 64
IMAGE_DIMS = (128, 128, 3)
MODEL = mymodels.allcnn.AllCNN
DESC = 'All CNN' #Short description of the training 
log =''
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

#Compiling
print("[INFO] compiling model...")
counter = Counter(train_generator.classes)                          

finalAct = 'softmax'
model = MODEL.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(counter),finalAct=finalAct)         
model.summary()

#Optimizer Config from Buda et al p.9
# INIT_LR = 0.1
# DECAY = 1e-4
# MOMENTUM = 0.9
# opt = SGD(lr=INIT_LR, momentum=MOMENTUM, decay=DECAY)
INIT_LR = 1e-6
DECAY = 0
beta_2 = 0.999
# opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=beta_2, decay=DECAY,
#  amsgrad=False,epsilon=1e-08)
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(opt,loss="categorical_crossentropy",metrics=["accuracy"])

if args.weight_init == 'lsuv':
    for x_batch, y_batch in train_generator:
        LSUV_init(model, x_batch)
        break # just one time


little_desc =args.desc
#Tensorboard 
tb_log_name  = little_desc+ str(MODEL) +'_lr:'+ str(INIT_LR) + '_' \
    + DESC.replace(' ','_') + 'Decay:'+str(DECAY) +'_'+ str(time.time())
tensorboard  = TensorBoard(log_dir="logs/{}".format( tb_log_name ))


# Checkpoint
if args.checkpoints:
    filepath=str(MODEL.__name__) +little_desc + \
        "_{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = os.path.sep.join(['checkpoints',filepath])
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                verbose=1, save_best_only=True, mode='max')
    callbacks_list = [tensorboard,checkpoint]
else: 
    callbacks_list = [tensorboard]


# 
print("[INFO] training network...")
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


fit_args = dict(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=callbacks_list)

#Class weights 
class_weighting = None
if args.weights :
    print('[INFO] Calculating class weights')
    counter = Counter(valid_generator.classes)                          
    max_val = float(max(counter.values()))
    log += 'Class weights were used\n'       
    class_weighting = {class_id : max_val/num_images for class_id, num_images in counter.items()} 
    fit_args['class_weight'] = class_weighting

#TRAIN
start = time.time()
H = model.fit_generator(**fit_args)
end = time.time()


# save the model to disk
print("[INFO] serializing network...")
model.save(args.model)


#Log creation 
now = datetime.datetime.now()
now_ =  now.strftime("%Y-%m-%d_%H:%M") 
PLOTS_DIR = 'plot_and_log'
plot_name = 'plot' + now_   
PLOT_FILE = os.path.sep.join([PLOTS_DIR,plot_name])
LOG_FILE = os.path.sep.join([PLOTS_DIR,'log.txt'])
log = '<---------------------------------------------------> \n'
log += DESC + '\n'
log += 'Image dims : ' + str(IMAGE_DIMS) + '\n'
log += 'Train batch size : ' + str(BS) + '\n'
log += 'Model name : ' + str(MODEL.__name__) + '\n'
log += 'Plot file name : ' + PLOT_FILE + '\n'
log += 'EPOCHS : ' + str(EPOCHS) + '\n'
log += 'Optimizer: ' + opt.__class__.__name__ +'\n'
log += 'Training time: ' +str( (end - start)/60) + ' min \n'
#write the log to file
with open(LOG_FILE,'a+') as f:
    f.write(log)


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



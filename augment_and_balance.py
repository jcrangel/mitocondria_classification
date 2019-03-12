#This will augment and balance the data (oversampling)
#That comes from the csv file 

#Use example :
#python augment_and_balance.py --column Classification --outdir aug_images2
#TODO: When min_double =2 , cancer = 2133, iPS~1512,etc not very balanced

import random
import cv2
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from config import data_config as config
import pandas as pd
from albumentations import (ElasticTransform,
Compose,HorizontalFlip,VerticalFlip,RandomRotate90,
GridDistortion,OpticalDistortion,RandomSizedCrop,OneOf)

def augment_and_save(aug, image,imageout_name='',ext='jpg'):
    image = aug(image=image)['image']
    if imageout_name == '' :
        imageout_name = str(random.randint(0,10000))+'_aug.' + ext 
        
    cv2.imwrite(imageout_name,image)
    
#Agument the given class in the given column a save the augmente images in
#images_dir_out
def augment_column_create_dir(csv_file, 
                              column,
                              class_to_augment,
                              images_dir_in, images_dir_out, batches=5):
    
    #Read file names from the csv 
    data = pd.read_csv(csv_file)
    criteria = data[column] == class_to_augment
    files_list=data[criteria]['image_filename'].values
    
    if not os.path.exists(images_dir_out):
        os.makedirs(images_dir_out)
        
    if not os.path.exists(images_dir_in):
        print('Images dir in doesn\'t  exist')
        return    
    for filename in files_list:
        img = cv2.imread(os.path.join(images_dir_in,filename))
        if img is not None: #File exist?
            #First just copy the file no augment
            original_file = os.path.sep.join([images_dir_out,filename])
            cv2.imwrite(original_file,img)
            
            s = filename.split('.')
            for i in range(0, batches): # Create the augmente files
                imageout_name = s[0]+'_'+str(random.randint(0,10000))+'_aug.'+s[1]
                imageout_name = os.path.sep.join([images_dir_out , imageout_name]) 
                augment_and_save(aug,img,imageout_name)
    

## 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outdir", required=True,
	help="path to output dir of augmented images")
ap.add_argument("-c", "--column", required=True,
	help="column to augment in the csv file")  
ap.add_argument("-m", "--min_doubling", required=False, default = 1,  
	help="Minimum number of batches for the biggest class. If min_doubling=2 the biggest\
        class will double. If min_doubling=1 the biggest class will just be copied")

args = ap.parse_args()


original_height, original_width = (614, 1024)
aug = Compose([
RandomSizedCrop(min_max_height=(500, 600), height=original_height, width=original_width, p=0.2),
VerticalFlip(p=0.5),
HorizontalFlip(p=0.5),    
RandomRotate90(p=0.5),
OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.06, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        #OpticalDistortion(p=0.5, distort_limit=1.7, shift_limit=0.5)                
        ], p=0.8)])



CSV_FILE = os.path.sep.join([config.BASE_PATH,config.TRAIN_FILE])
images_dir_in = os.path.sep.join([config.BASE_PATH,config.CROP_IMAGES])

column = args.column

if column == 'Classification' or column == 'Morphology': 
    banned_classes= ['other']
if column == 'Lifespan':
    banned_classes= ['unknown']


data = pd.read_csv(CSV_FILE)
classes = data[column].unique()
classes_to_augment = data[column].value_counts()
classes_to_augment = classes_to_augment.to_dict()
map(classes_to_augment.pop,banned_classes)
maximum = max(classes_to_augment.values()) 
maximum = maximum * int(args.min_doubling)


#Creating the training set

for class_,size in classes_to_augment.iteritems():
    print('Augmenting class: '+ class_)
    images_dir_out = os.path.sep.join([config.BASE_PATH,args.outdir,column,
    'train',class_ ])
    batches = int(maximum/size)
    if batches <= 1: #When we just copy the images
        batches = 0 
    augment_column_create_dir(CSV_FILE, 
                              column,
                              class_,
                              images_dir_in, images_dir_out, batches)


#Creating the validation set, just copy images
CSV_FILE = os.path.sep.join([config.BASE_PATH,config.VAL_FILE])
# data = pd.read_csv(CSV_FILE)
# classes_to_augment = data[column].value_counts()
#Copy validation set from csv
batches = 0 # Just copy, don't augment
for (class_,values) in classes_to_augment.iteritems():
    print('Copying '+ class_ + ' with ' + str(values) + ' images...')
    images_dir_out = os.path.sep.join([config.BASE_PATH,args.outdir,column,'validation',class_])
    augment_column_create_dir(CSV_FILE, 
                                  column,
                                  class_,
                                  images_dir_in, images_dir_out, batches)
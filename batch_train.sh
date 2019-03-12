#!/bin/bash
python train_dir.py  --column Morphology --dataset cell_image/aug_images2 --model cell_Classification.model --checkpoints --desc MorphologyAug2 
python train_dir.py  --column Lifespan --dataset cell_image/aug_images2 --model cell_Classification.model --checkpoints --desc LifespanAug2 
# galaxy-image-classification
Classification of galaxy images. 
1. The Galaxy Zoo 2 dataset was downloaded from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data?select=images_training_rev1.zip 
2. The dataset contains JPG images of 61578 galaxies. Files are named according to their GalaxyId.
3. This workflow aims at referencing the work done by paper: https://link.springer.com/article/10.1007/s10509-019-3540-1
4. The workflow mainly comprises of DataLabel.py file which creates a subset of images needed for the classification task from the original Galaxy Zoo 2.
5. The following jobs of the workflow perform image resizing and data augmentation.

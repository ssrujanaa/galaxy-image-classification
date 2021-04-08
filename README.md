# galaxy-image-classification
Classification of galaxy images. 
1. The Galaxy Zoo 2 dataset was downloaded from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data?select=images_training_rev1.zip 
2. The dataset contains JPG images of 61578 galaxies. Files are named according to their GalaxyId.
3. This workflow aims at referencing the work done by paper: https://link.springer.com/article/10.1007/s10509-019-3540-1
4. The workflow mainly comprises of the following python scripts:
    1. Data_Labeling.py -> This script is mainly focuses on getting a smaller subset of the Galaxy Zoo 2 dataset that matches either of the 5 different target morphologies as mentioned in the paper.
    
    2. Following this, data partioning into train, test and val is performed. The partitioned data is used for preprocessing.
 
    3. Training_images_Preprocess1.py -> This script performs random cropping (Scale jittering) and image resizing. The images are        
    converted and store as numpy arrays in the 'Training_images_Preprocess1.npy' and 'Training_labels_Preprocess1.npy' files.
    
    4. Training_images_Preprocess2.py -> This script performs Random crop, flip, rotation, Brightness, Contrast, Hue and                  
    Saturation and stores as numpy arrays in the 'Training_images_Preprocess2.npy' and 'Training_labels_Preprocess2.npy' files.
    
    5. Val_Preprocess.py and Test_Preprocess.py -> These scripts perform random cropping (Scale jittering) and image resizing on          
    val and test images respectively.
    
    6. Workflow.ipynb -> Workflow script to run all the above mentioned jobs using the Pegasus API.


Alternatively, I have another subfolder called "WithoutPegasus". This folder has all the work that can be run on local machine without using Pegasus. To run the entire model, please follow the following steps:
1. First, download the dataset from kaggle: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data?select=images_training_rev1.zip 
2. Run the Data_Labeling.py script. This file is responsible for getting a smaller subset of the Galaxy Zoo 2 dataset that matches either of the 5 different target morphologies as mentioned in the paper.
3. Next, run the Data_Partition.py script. Data partioning into train, test and val is performed.
4. To preprocess train dataset, run the Training_Preprocess1.py followed by Training_Preprocess2.py
5. To preprocess test datatset, run the Test_Preprocess.py
6. To preprocess val dataset, run the Val_Preprocess.py
7. Finally, run the train.py script to perform training on the preprocessed data. 
    

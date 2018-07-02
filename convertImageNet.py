

#We should get a glob to get all the image files we want to examine
#We should get a dictionary to map file paths from the glob to indexes
#We should read the text files to generate labels for these images_file
#then we should load the images. When we reach a max value we should write to a binary file
from glob import glob
import os
import numpy as np
import loadFeedDict
from PIL import Image
import saveBinary
from random import shuffle
np.set_printoptions(threshold=np.nan)
IMAGENET_PATH = '/Data/Imagenet/imagenet_object_detection_train/ILSVRC/Data/DET/train/'
LABELS_PATH = '/Data/Imagenet/imagenet_object_detection_train/ILSVRC/ImageSets/DET/train_'
TRAIN_INPUT_SAVE = '/Data/Imagenet/train_images'
TRAIN_LABEL_SAVE = '/Data/Imagenet/train_labels'
HEIGHT, WIDTH = 512, 512
batch_size= 1000
I_2013 = glob(os.path.join(IMAGENET_PATH,'ILSVRC2013_train','*','*'))
I_2014 = glob(os.path.join(IMAGENET_PATH,'ILSVRC2014*','*'))
I_2013_extra = glob(os.path.join(IMAGENET_PATH,'ILSVRC2013_train_*','*'))
NUM_CLASSES = 200
ALL_FILES_FULL = I_2013 + I_2014 + I_2013_extra
shuffle(ALL_FILES_FULL)
ALL_FILES = [f.replace(IMAGENET_PATH, '').replace('.JPEG', '') for f in ALL_FILES_FULL]
files_dict = {v: i for i, v in enumerate(ALL_FILES)}
labels = np.full((len(ALL_FILES), NUM_CLASSES), -1, dtype = np.int8)

for i in range(NUM_CLASSES):
    txt_path = LABELS_PATH + str(i+1) + '.txt'
    print(i)
    with open(txt_path, 'r') as f:
        values = [line.replace('\n','').split(' ') for line in f]
        for val in values:
            # print(val[0])
            # print(files_dict[val[0]])
            labels[files_dict[val[0]],i] = int(val[1])
i = 0
while i != -1:
    if i+batch_size <= len(labels):
        j = i + batch_size
    else:
        break
        j = -1
    print(i)

    images = loadFeedDict.FromJPEG.get_batch(ALL_FILES_FULL[i:j], WIDTH, HEIGHT)
    labels_ = labels[i:j]
    # print(images[0])
    # image_ex = Image.fromarray(images[0])
    # # print(np.argmax(labels[i]))
    # image_ex.show()
    # print('mid')
    saveBinary.append_binary_file(TRAIN_INPUT_SAVE,images.tobytes())
    saveBinary.append_binary_file(TRAIN_LABEL_SAVE,labels_.tobytes())
    i = j

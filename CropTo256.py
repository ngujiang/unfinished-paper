import cv2
import numpy as np
import random
import os

from os import listdir
from os.path import join
import  tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images
path_1 = "/data/ljy/cycleGan-1-12/data2/ir/"              
img_path_1 = list_images(path_1)

i = 0
num=0
for p_1 in img_path_1:
    i = i + 1
    print(i)
    img_1 = cv2.imread(p_1)
    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    output1="/data/ljy/stargan/dataset/celebA/train/"
    path_output_1 = output1+"ir"+str(i) + ".jpg"
    cv2.imwrite(path_output_1,img_gray_1)


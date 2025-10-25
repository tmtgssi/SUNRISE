import cv2
import numpy as np
import os
typs=['test','valid','train']
for typ in typs:
    
    path='/home/pal/Documents/cvpr2024/WGSR_saffron/classical_SR_datasets/saffron/'+typ+'/images/'
    newpath='/home/pal/Documents/cvpr2024/WGSR_saffron/classical_SR_datasets/saffron/'+typ+'/images_lr/'
    for name in os.listdir(path):
        print(path+name)
        img = cv2.imread(path+name)
        height, width, channels = img.shape
        print(height, width, channels)
        res = cv2.resize(img, (int(height/4), int(width/4)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(newpath+name, res)


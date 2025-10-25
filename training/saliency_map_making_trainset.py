import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
path='../classical_SR_datasets/DIV2K_train_HR/'
pathnew='../classical_SR_datasets/DIV2K_train_HR_saliency/'
for name in os.listdir(path):
    img = cv2.imread(path+name)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_component = lab[:,:,1]
    th = cv2.threshold(a_component,140,255,cv2.THRESH_BINARY)[1]
    blur = cv2.GaussianBlur(th,(13,13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    cv2.imwrite(pathnew+name,super_imposed_img)


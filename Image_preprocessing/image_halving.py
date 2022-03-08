import numpy as np
try:
    import cv2
except:
    print('to run this script you will need to install OpenCV library')
    print('ending execution')
    import cv2
import os

path = input('enter the path where your images are')
print('/n')
outpath = input('enter the path where you want to save the cropped images')
#path = '/Users/cnieto/Downloads/fets'
xrays = os.listdir(path)
print(xrays)




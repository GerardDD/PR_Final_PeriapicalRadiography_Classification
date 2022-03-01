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
outpath = input('enter the path where you want to save the censored images')
# path = '/Users/cnieto/Downloads/fets'
xrays = os.listdir(path)
print(xrays)

for i in xrays:
    image = cv2.imread(f'{path}/{i}')
    # extract 1/10 of the height
    ten = int(image.shape[0] /10)
    # extract 1/3 of the width
    three = int(image.shape[1] / 3)
    #create black rectangle on top left
    cv2.rectangle(image, (0,0),(three,ten),(0,0,0),-1)
    #create black rectangle on top right
    cv2.rectangle(image, (image.shape[1],0),(image.shape[1] - three,ten),(0,0,0),-1)
    cv2.imwrite(f'{outpath}/{i[:-4]}_cens.jpg', image)
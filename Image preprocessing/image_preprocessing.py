import numpy as np
import cv2
import os

image = cv2.imread('/Users/cnieto/Downloads/fets/4_age34_sexF.png')
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))

#cv2.imshow('test',image)
#cv2.waitKey(0)
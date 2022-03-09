import numpy as np
try:
    import cv2
except:
    print('to run this script you will need to install OpenCV library')
    print('ending execution')
    import cv2
import os

#path = input('enter the path where your images are')
#print('/n')
#outpath = input('enter the path where you want to save the cropped images')
#path = '/Users/cnieto/Downloads/fets'
#xrays = os.listdir(path)
#print(xrays)

image = cv2.imread('/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image_preprocessing/con_imagen/20_age59_sexF_cens.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("Image", image)
cv2.waitKey(0)
#
thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)
cv2.waitKey(0)
#
thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)

masked = cv2.bitwise_and(image,image,mask = thresh)
cv2.imshow("masked",masked)
cv2.waitKey(0)



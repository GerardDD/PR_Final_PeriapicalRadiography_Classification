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

#image = cv2.imread('/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image_preprocessing/con_imagen/20_age59_sexF_cens.jpg')
for i in xrays:
    try:

        image = cv2.imread(f'{path}/{i}')
        # extract 1/4 of the width
        proportion_x = int(image.shape[1] / 4)
        proportion_y = int(image.shape[0] / 2)
    # crop image 
        cropped = image[proportion_y: , proportion_x:-proportion_x] # notation is weird: [starty:endy, startx:endx]
        cv2.imwrite(f'{outpath}/{i[:-4]}_cropped.jpg', cropped)
    except:
        pass



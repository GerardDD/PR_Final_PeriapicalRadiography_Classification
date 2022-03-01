import numpy as np
try:
    import cv2
except:
    print(' to run this script you will need to install OpenCV library')
import os

# path = input('enter the path where your images are')
path = '/Users/cnieto/Downloads/fets'
xrays = os.listdir(path)
print(xrays)

for i in xrays:
    image = cv2.imread(f'{path}/{i}')
    cv2.imshow('original',image)
    print("i'm in image")
    # extract 1/10 of the height
    ten = int(image.shape[0] /10)
    # extract 1/3 of the width
    three = int(image.shape[1] / 3)
    #create black rectangle on top left
    cv2.rectangle(image, (0,0),(three,ten),(0,0,0),-1)
    #create black rectangle on top right
    cv2.rectangle(image, (image.shape[1],0),(image.shape[1] - three,ten),(0,0,0),-1)
    print(f'/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image preprocessing/testing/{i[:-4]}_cens.jpg')
    cv2.imwrite(f'/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image preprocessing/testing/{i[:-4]}_cens.jpg', image)



#image = cv2.imread('/Users/cnieto/Downloads/fets/4_age34_sexF.png')
#print("width: {} pixels".format(image.shape[1]))
#print("height: {} pixels".format(image.shape[0]))
#
## extract 1/6 of the height
#ten = int(image.shape[0] /10)
#
## extract 1/3 of the width
#three = int(image.shape[1] / 3)
#
## create black rectancle on top left
#cv2.imshow('original',image)
#cv2.waitKey(0)
#cv2.rectangle(image, (0,0),(three,ten),(0,0,0),-1)
#
#cv2.imshow('Censored version',image)
#cv2.waitKey(0)
#
## create black rectancle on top right
#
#cv2.rectangle(image, (image.shape[1],0),(image.shape[1] - three,ten),(0,0,0),-1)
#cv2.imshow('Censored version',image)
#cv2.waitKey(0)
#
#cv2.imwrite("/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image preprocessing/testing/4_age34_sexF.jpg", image)

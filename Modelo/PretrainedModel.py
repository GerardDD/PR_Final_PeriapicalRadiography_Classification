from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import vgg16
import numpy as np
import cv2
import os
import pickle

path = input('enter the path where your images are')
xrays = os.listdir(path)

print("[INFO] loading network...")
model = vgg16.VGG16(weights="imagenet")

print("[INFO] loading and preprocessing images...")

vectorized = []

for i in xrays:
    try:
        print(f'{path}/{i}')
        image = image_utils.load_img(f'{path}/{i}', target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        vectorized.append(model.predict(image))
    except:
        print(f'{path}/{i} skipped')

f = open('/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image_preprocessing/sin_imagen_pickle.txt', 'wb')
pickle.dump(vectorized, file=f)


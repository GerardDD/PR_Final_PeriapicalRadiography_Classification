from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import vgg16
import numpy as np
import cv2
import os
import pickle
import regex as re

#path = input('enter the path where your images are')
#pickle_path = input('enter the pickle file that must be overwritten')
path = '/Users/cnieto/IronHack/Personal_projects/sin_imagen_croppinghalf'
pickle_path = '/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image_preprocessing/sin_imagen_noCens.txt'
xrays = os.listdir(path)

print("[INFO] loading network...")
model = vgg16.VGG16(weights="imagenet")

print("[INFO] loading and preprocessing images...")

vectorized = []
sex_patt = re.compile('sex(F|M|U)')
age_patt = re.compile('age(\d*)_')

for i in xrays:
    try:
        print(f'{path}/{i}')
        age = age_patt.findall(i)[0]
        print(age)
        sex = sex_patt.findall(i)[0]
        print(sex)
        image = image_utils.load_img(f'{path}/{i}', target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        vector = model.predict(image)
        vector_age = np.append(vector,age)
        vector_sex = np.append(vector_age,sex)
        vectorized.append(vector_sex)
    except:
        print(f'{path}/{i} skipped')

f = open(pickle_path, 'wb')
pickle.dump(vectorized, file=f)


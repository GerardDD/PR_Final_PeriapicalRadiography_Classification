import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import  vgg19
import os

# import trained model

fmodel = pickle.load(open('./final_model.sav','rb'))

# load images

path = input('enter the folder where your unclassified images are\n')
xrays = os.listdir(path)

# vectorized unclassified images

print("[INFO] loading network...")
model = vgg19.VGG19(weights="imagenet")
print("[INFO] loading and preprocessing images...")
vectorized = []
image_tag = []
for i in xrays:
    try:
        print(f'vectorizing {path}/{i} ...')
        image = image_utils.load_img(f'{path}/{i}', target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        vector = model.predict(image)
        vectorized.append(vector)
        image_tag.append(i)

    except:
        print(f'{path}/{i} skipped')

# dump images onto target dataset

df = pd.DataFrame(
    columns= range(0,len(vectorized[0][0])),
    index= range(0,len(vectorized)))

pos = -1
for i in vectorized:
    for j in i:
        col = 0
        pos += 1
        for z in j:
            df.iloc[pos,col] = z
            col += 1

df_pred = pd.DataFrame(
    columns = range(0,2),
    index = range(0,len(xrays))
    )

# pred values:
for i,j in zip(df.index,image_tag):
    result = fmodel.predict(df.loc[[i,]])[0]
    if result == 0:
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred.loc[i,0] = j
    df_pred.loc[i,1] = result

# export excel with results 

df_pred.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred.to_csv('./predictions.csv')
df_pred.to_excel('./predictions.xls')
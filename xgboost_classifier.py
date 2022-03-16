from statistics import mode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import KBinsDiscretizer
import pickle
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import  vgg19
import os
import regex as re

modelo_xgboost = xgb.Booster()
modelo_xgboost.load_model("./Modelo/modelo_xgboost.model")

# load images

#path = input('enter the folder where your unclassified images are\n')
path = '/Users/cnieto/Downloads/prueba/prueba_F'
xrays = os.listdir(path)

# vectorized unclassified images

print("[INFO] loading network...")
model = vgg19.VGG19(weights="imagenet")
print("[INFO] loading and preprocessing images...")
vectorized = []
image_tag = []
sex_patt = re.compile('sex(F|M|U)')
age_patt = re.compile('age(\d*)_')
for i in xrays:
    try:
        print(f'vectorizing {path}/{i} ...')
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
        image_tag.append(i)

    except:
        print(f'{path}/{i} skipped')

# dump images onto target dataset

df = pd.DataFrame(
    columns= range(0,len(vectorized[0])),
    index= range(0,len(vectorized)))

pos = -1
for i in vectorized:
    pos += 1
    col = 0
    for j in i:
        df.iloc[pos,col] = j
        col += 1
            

df.to_csv('/Users/cnieto/Downloads/prueba/df_prueba/df_prueba.csv')

if df.iloc[0,1001] == 'F':
    df['sex_0'] = 1
    df['sex_1'] = 0

for i,j in df.iterrows():
    if int(df.iloc[i,1000]) < 15:
        df.loc[i,'age_bin'] = 0
    elif int(df.iloc[i,1000]) < 30:
        df.loc[i,'age_bin'] = 1
    elif int(df.iloc[i,1000]) < 45:
        df.loc[i,'age_bin'] = 2
    elif int(df.iloc[i,1000]) < 60:
        df.loc[i,'age_bin'] = 3
    else:
        df.loc[i,'age_bin'] = 4

df.drop([1000,1001],axis=1,inplace=True)




df_pred = pd.DataFrame(
    columns = range(0,2),
    index = range(0,len(xrays))
    )

# transform vectorized df to xgboost matrix

#df_m_prueba2 = df.drop(['sex','age'],axis=1)
df_m_prueba2 = df.apply(pd.to_numeric)

unclassified =  xgb.DMatrix(df_m_prueba2)
pred = modelo_xgboost.predict(unclassified)

# pred values:
for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.5:
        result = 0
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        result = 1
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred.loc[i,0] = j
    df_pred.loc[i,1] = result

# export excel with results 

df_pred.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred.to_csv('./predictions.csv')
df_pred.to_excel('./predictions.xls')


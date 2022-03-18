from statistics import mode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import KBinsDiscretizer
import pickle
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import  vgg19
import os
import regex as re

# loading the 3 models

# female

modelo_ada_f = pickle.load(open('./Modelo/modelo_adaboost_as_f.sav','rb'))


# male

modelo_ada_m = pickle.load(open('./Modelo/modelo_adaboost_as_m.sav','rb'))

# mixed

modelo_ada_full = pickle.load(open('./Modelo/modelo_adaboost_as_full.sav','rb'))

# load images

#path = input('enter the folder where your unclassified images are\n')
path = '/Users/cnieto/Downloads/2/2_prueba'
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
else:
    df['sex_0'] = 0
    df['sex_1'] = 1

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

df_pred_m = pd.DataFrame(
    columns = range(0,2),
    index = range(0,len(xrays))
    )

df_pred_full = pd.DataFrame(
    columns = range(0,2),
    index = range(0,len(xrays))
    )



# pred values:

pred = modelo_ada_f.predict(df)
print('model based on female images\n')
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
df_pred.to_csv('./predictions_ada_f.csv')
df_pred.to_excel('./predictions_ada_f.xls')


# pred values:

pred = modelo_ada_m.predict(df)
print('model based on male images\n')
for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.5:
        result = 0
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        result = 1
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred_m.loc[i,0] = j
    df_pred_m.loc[i,1] = result

# export excel with results 

df_pred_m.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred_m.to_csv('./predictions_ada_m.csv')
df_pred_m.to_excel('./predictions_ada_m.xls')

# pred values:

pred = modelo_ada_full.predict(df)
print('model based on mixed images\n')
for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.5:
        result = 0
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        result = 1
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred_full.loc[i,0] = j
    df_pred_full.loc[i,1] = result

# export excel with results 

df_pred_full.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred_full.to_csv('./predictions_ada_full.csv')
df_pred_full.to_excel('./predictions_ada_full.xls')
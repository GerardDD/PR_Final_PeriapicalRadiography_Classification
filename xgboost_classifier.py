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

# loading the 3 models

# female
modelo_xgboost = xgb.Booster()
modelo_xgboost.load_model("./Modelo/modelo_xgboost.model")

# male

modelo_xgboost_m = xgb.Booster()
modelo_xgboost_m.load_model("./Modelo/modelo_xgboost_m.model")

# mixed

modelo_xgboost_full = xgb.Booster()
modelo_xgboost_full.load_model("./Modelo/modelo_xgboost_full.model")




# load images

path = input('enter the folder where your unclassified images are\n')
#path = '/Users/cnieto/Downloads/Desktop/sin_im'
#path = '/Users/cnieto/IronHack/Personal_projects/con_imagen_noCens'
#path = '/Users/cnieto/Desktop/set_alumni'
#path = '/Users/cnieto/Downloads/2/2_prueba'
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

# transform vectorized df to xgboost matrix

#df_m_prueba2 = df.drop(['sex','age'],axis=1)
df_m_prueba2 = df.apply(pd.to_numeric)

unclassified =  xgb.DMatrix(df_m_prueba2)
pred = modelo_xgboost.predict(unclassified)

# pred values:
print('model based on female images\n')
for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.48:
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


# Male - pred values:
print('\nmodel based on male images\n')
pred = modelo_xgboost_m.predict(unclassified)

for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.48:
        result = 0
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        result = 1
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred_m.loc[i,0] = j
    df_pred_m.loc[i,1] = result

# export excel with results 

df_pred_m.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred_m.to_csv('./predictions_m.csv')
df_pred_m.to_excel('./predictions_m.xls')

# Full - pred values:

print('\nmodel based on mixed images\n')
pred = modelo_xgboost_full.predict(unclassified)

for i,j in zip(df.index,image_tag):
    result = pred[i]
    print(f'result is {result}')
    if result < 0.48:
        result = 0
        print(f'NO unhealthy tooth detected in {j}')
        
    else:
        result = 1
        print(f'UNHEALTHY tooth detected in {j}')
    df_pred_full.loc[i,0] = j
    df_pred_full.loc[i,1] = result

# export excel with results 

df_pred_full.rename(columns={0:'image name',1:'result'}, inplace=True)
df_pred_full.to_csv('./predictions_full.csv')
df_pred_full.to_excel('./predictions_full.xls')

df_combi = df_pred.copy()

df_combi['image name m'] = df_pred_m['image name']
df_combi['result m'] = df_pred_m['result']
df_combi2 = df_combi.copy()
df_combi['image name full'] = df_pred_full['image name']
df_combi['result full'] = df_pred_full['result']
df_combi['combi'] = df_combi['result'] + df_combi['result m'] + df_combi['result full']
df_combi2['combi'] = df_combi2['result'] + df_combi2['result m']
df_combi['Veredict'] = df_combi['combi'].apply(lambda x: 'Healthy' if x < 2 else 'Unealthy')
df_combi2['Veredict'] = df_combi2['combi'].apply(lambda x: 'Healthy' if x < 1 else 'Unealthy')
df_combi['short_name'] = df_combi['image name'].apply(lambda x: 'SIN' if x.__contains__("SIN") else "CON" )
df_combi2['short_name'] = df_combi2['image name'].apply(lambda x: 'SIN' if x.__contains__("SIN") else "CON" )

for i,j in df_combi.short_name.iteritems():
    if j == 'SIN' and df_combi.loc[i,'combi'] < 2:
        df_combi.loc[i,'check'] = 'TRUE'
    elif j == 'CON' and df_combi.loc[i,'combi'] > 1:
        df_combi.loc[i,'check'] = 'TRUE'
    else:
        df_combi.loc[i,'check'] = 'FALSE'

for i,j in df_combi2.short_name.iteritems():
    if j == 'SIN' and df_combi2.loc[i,'combi'] <= 1:
        df_combi2.loc[i,'check'] = 'TRUE'
    elif j == 'CON' and df_combi2.loc[i,'combi'] > 1:
        df_combi2.loc[i,'check'] = 'TRUE'
    else:
        df_combi2.loc[i,'check'] = 'FALSE'


df_combi.to_excel('./predictions_combi.xls')
df_combi[['image name','Veredict','check']].to_excel('./predictions_combi_simple.xls')

df_combi2.to_excel('./predictions_combi2.xls')
df_combi2[['image name','Veredict','check']].to_excel('./predictions_combi2_simple.xls')
from statistics import mode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestCentroid
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import KBinsDiscretizer
import pickle

# importamos df training

df = pd.read_csv('dataset_selected_as.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.rename(columns={'1000':'age', '1001':'sex'},inplace=True)
df_origin = df.copy()
#df.drop(df[(df['age'] == 999) | (df['sex'] == 'U')].index, axis=0,inplace=True)

encoder = ce.BinaryEncoder()
df['sex_0'] = encoder.fit_transform(df['sex'])['sex_0']
df['sex_1'] = encoder.fit_transform(df['sex'])['sex_1']

kdisc = KBinsDiscretizer(n_bins=5, encode='ordinal')
df['age_bin'] = kdisc.fit_transform(df[['age']])

# df de prueba

prueba_arr = pickle.load(open('/Users/cnieto/IronHack/Personal_projects/PR_Final_PeriapicalRadiography_Classification/Image_preprocessing/real_test_as.txt','rb'))
df_prueba = pd.DataFrame(
    columns= range(0,len(prueba_arr[0])),
    index= range(0,len(prueba_arr)))

pos = -1
for i in prueba_arr:
    pos += 1
    col = 0
    for j in i:
        try:
            df_prueba.iloc[pos,col] = float(j)
            col += 1
        except:
            df_prueba.iloc[pos,col] = j
            col += 1

df_prueba.rename(columns={1000:'age', 1001:'sex'},inplace=True)
df_prueba['sex_0'] = encoder.fit_transform(df_prueba['sex'])['sex_0']
df_prueba['sex_1'] = encoder.fit_transform(df_prueba['sex'])['sex_1']
kdisc = KBinsDiscretizer(n_bins=5, encode='ordinal')
df_prueba['age_bin'] = kdisc.fit_transform(df_prueba[['age']])

df_m_prueba = df_prueba[df_prueba.sex =='F']


# only female

df_f = df[df.sex == 'F']

# xgboosting

X_train, X_test = train_test_split(df_f,test_size=0.15)

X_train_mat = xgb.DMatrix(X_train.drop(['Target','sex','age'],axis=1),label=X_train["Target"])
X_test_mat = xgb.DMatrix(X_test.drop(['Target','sex','age'],axis=1),label=X_test["Target"])

#parametros = {"booster":"gbtree", "max_depth": 2000, "eta": 0.01, "objective": "binary:logistic", "nthread":2}
parametros = {"booster":"dart", "max_depth": 2000, "eta": 0.1, "objective": "binary:logistic", "nthread":2}

rondas = 10

evaluacion = [(X_test_mat, "eval"), (X_train_mat, "train")]

modelo = xgb.train(parametros, X_train_mat, rondas, evaluacion)
#modelo.save_model("./modelo_xgboost.model")


prediccion = modelo.predict(X_test_mat)
prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)

# evaluacion del modelo

def get_metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = get_metricas(X_test["Target"], prediccion)
[print(i) for i in metricas]

#parametros_02 = {"booster":"gbtree", "max_depth": 4, "eta": .3, "objective": "binary:logistic", "nthread":2}
#rondas_02 = 100
#modelo_02 = xgb.train(parametros_02, X_train_mat, rondas_02, evaluacion, early_stopping_rounds=10)
#
#prediccion_02 = modelo_02.predict(X_test_mat)
#prediccion_02 = [1 if i > .5 else 0 for i in prediccion_02]
#metricas_02 = get_metricas(X_test["Target"], prediccion_02)
#[print(i) for i in metricas_02]
#
#parametros_03 = {"booster":"gbtree", "max_depth": 6, "eta": .2, "objective": "binary:logistic", "nthread":2}
#rondas_03 = 100
#modelo_03 = xgb.train(parametros_02, X_train_mat, rondas_02, evaluacion, early_stopping_rounds=10)
#
#prediccion_03 = modelo_03.predict(X_test_mat)
#prediccion_03 = [1 if i > .5 else 0 for i in prediccion_03]
#metricas_03 = get_metricas(X_test["Target"], prediccion_03)
#[print(i) for i in metricas_03]

# model df_m

df_m = df[df.sex == 'M']

X_train, X_test = train_test_split(df_m,test_size=0.1)

X_train_mat = xgb.DMatrix(X_train.drop(['Target','sex','age'],axis=1),label=X_train["Target"])
X_test_mat = xgb.DMatrix(X_test.drop(['Target','sex','age'],axis=1),label=X_test["Target"])

parametros = {"booster":"gbtree", "max_depth":1500, "eta": 0.1, "objective": "binary:logistic", "nthread":2}
#parametros = {"booster":"dart", "max_depth": 1500, "eta": 0.2, "objective": "binary:logistic", "nthread":2}

rondas = 30

evaluacion = [(X_test_mat, "eval"), (X_train_mat, "train")]

modelo = xgb.train(parametros, X_train_mat, rondas, evaluacion)
#modelo.save_model("./modelo_xgboost_m.model")

prediccion = modelo.predict(X_test_mat)
prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)

metricas = get_metricas(X_test["Target"], prediccion)
[print(i) for i in metricas]


# prueba con full dataset

X_train, X_test = train_test_split(df,test_size=0.1)

X_train_mat = xgb.DMatrix(X_train.drop(['Target','sex','age'],axis=1),label=X_train["Target"])
X_test_mat = xgb.DMatrix(X_test.drop(['Target','sex','age'],axis=1),label=X_test["Target"])

parametros = {"booster":"gbtree", "max_depth": 2000, "eta": 0.01, "objective": "binary:logistic", "nthread":2}
#parametros = {"booster":"dart", "max_depth": 2000, "eta": 0.1, "objective": "binary:logistic", "nthread":2}

rondas = 10

evaluacion = [(X_test_mat, "eval"), (X_train_mat, "train")]

modelo = xgb.train(parametros, X_train_mat, rondas, evaluacion)
#modelo.save_model("./modelo_xgboost_full.model")

prediccion = modelo.predict(X_test_mat)
prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)

metricas = get_metricas(X_test["Target"], prediccion)
[print(i) for i in metricas]


# prueba real

modelo_importado = xgb.Booster()
modelo_importado.load_model("./modelo_xgboost.model")

df_m_prueba2 = df_m_prueba.drop(['sex','age'],axis=1)
df_m_prueba2 = df_m_prueba2.apply(pd.to_numeric)

unclassified =  xgb.DMatrix(df_m_prueba2)

prediccion = modelo_importado.predict(unclassified)
prediccion = [1 if i > .5 else 0 for i in prediccion]
print(prediccion)


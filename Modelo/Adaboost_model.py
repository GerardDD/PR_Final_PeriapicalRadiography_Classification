#Tuned Hyperparameters : {'classifier__learning_rate': 0.025,
#  'classifier__n_estimators': 150}
#Accuracy : 0.615934065934066

from statistics import mode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import NearestCentroid
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
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

# adaboost
X = df_f.drop(['Target','sex','age'],axis=1)
y = df_f.Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)


#parametros = {"booster":"gbtree", "max_depth": 2000, "eta": 0.01, "objective": "binary:logistic", "nthread":2}
#parametros = {
 # 'n_estimators': 150,
  # 'learning_rate': 0.025,
  # 'algorithm':'SAMME'

 #  }

modelo = AdaBoostClassifier(n_estimators=150, learning_rate=0.025)

modelo.fit(X_train,y_train)
pickle.dump(modelo , file = open('modelo_adaboost_as_f.sav','wb'))
#modelo.save_model("./modelo_adaboost_as_f.model")


prediccion = modelo.predict(X_test)
#prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)

# evaluacion del modelo

def get_metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = get_metricas(y_test, prediccion)
[print(i) for i in metricas]

# only male

df_f = df[df.sex == 'M']

# adaboost
X = df_f.drop(['Target','sex','age'],axis=1)
y = df_f.Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

modelo = AdaBoostClassifier(n_estimators=150, learning_rate=0.025)

modelo.fit(X_train,y_train)
pickle.dump(modelo , file = open('modelo_adaboost_as_m.sav','wb'))
#modelo.save_model("./modelo_adaboost_as_f.model")


prediccion = modelo.predict(X_test)
#prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)
metricas = get_metricas(y_test, prediccion)
[print(i) for i in metricas]


# mixed df

df_f = df.copy()

# adaboost
X = df_f.drop(['Target','sex','age'],axis=1)
y = df_f.Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

modelo = AdaBoostClassifier(n_estimators=150, learning_rate=0.025)

modelo.fit(X_train,y_train)
pickle.dump(modelo , file = open('modelo_adaboost_as_full.sav','wb'))
#modelo m.save_model("./modelo_adaboost_as_f.model")


prediccion = modelo.predict(X_test)
#prediccion = [1 if i > .5 else 0 for i in prediccion]

print(prediccion)
metricas = get_metricas(y_test, prediccion)
[print(i) for i in metricas]


#####


# prueba real

#modelo_importado = xgb.Booster()
#modelo_importado.load_model("./modelo_xgboost.model")
#
#df_m_prueba2 = df_m_prueba.drop(['sex','age'],axis=1)
#df_m_prueba2 = df_m_prueba2.apply(pd.to_numeric)
#
#unclassified =  xgb.DMatrix(df_m_prueba2)
#
#prediccion = modelo_importado.predict(unclassified)
#prediccion = [1 if i > .5 else 0 for i in prediccion]
#print(prediccion)


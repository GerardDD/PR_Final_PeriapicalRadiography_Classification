import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestCentroid
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('df_unCens.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.rename(columns={'1000':'age', '1001':'sex'},inplace=True)
encoder = ce.BinaryEncoder()
df['sex_0'] = encoder.fit_transform(df['sex'])['sex_0']
df['sex_1'] = encoder.fit_transform(df['sex'])['sex_1']
X = df.drop(['Target','age','sex_0','sex_1','sex'],axis=1)
y = df.Target

classifiers = [
    NuSVC(), 
    NearestCentroid(), 
    GaussianNB(),
    RandomForestClassifier(),
    ExtraTreesClassifier()
]

# parameter grids for the various classifiers
NuSVC_parameters = {
    'classifier__nu' : [0.3,0.5,0.7],
    'classifier__kernel' : ['linear','poly','rbg','sigmoid'],
    'classifier__gamma':['scale','auto']
}
NearestCentroid_parameters = {
    'classifier__shrink_threshold' : [None]
    
}
GaussianNB_parameters = {
    'classifier__var_smoothing': [1e-9,1e-8,1e-7]
    
}
RandomForest_parameters = {
    'classifier__n_estimators': [50,100,150,200],
    'classifier__criterion' : ['gini','entropy']
}
ExtraTree_parameters = {
    'classifier__n_estimators': [10,100,1000],
    'classifier__criterion' : ['gini','entropy']
}

parameters = [
    NuSVC_parameters,
    NearestCentroid_parameters,
    GaussianNB_parameters,
    RandomForest_parameters,
    ExtraTree_parameters

]

estimators = []
# iterate through each classifier and use GridSearchCV
grid = 4
if grid == 1:

    for i, classifier in enumerate(classifiers):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X, y)
        print("Tuned Hyperparameters :", clf.best_params_)
        print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
        #clf.fit(X_5_train, y_5_train)
        #print("Tuned Hyperparameters :", clf.best_params_)
        #print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
    print(estimators)
elif grid == 2:
    print('grid2')
    df_m = df[df.sex == 'M']
    X = df_m.drop(['Target','age','sex_0','sex_1','sex'],axis=1)
    y = df_m.Target
    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X, y)
        print("Tuned Hyperparameters :", clf.best_params_)
        print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
        #clf.fit(X_5_train, y_5_train)
        #print("Tuned Hyperparameters :", clf.best_params_)
        #print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
    print(estimators) 
elif grid == 3:
    df_m = df[df.sex == 'F']
    X = df_m.drop(['Target','age','sex_0','sex_1','sex'],axis=1)
    y = df_m.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X, y)
        print("Tuned Hyperparameters :", clf.best_params_)
        print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
        #clf.fit(X_5_train, y_5_train)
        #print("Tuned Hyperparameters :", clf.best_params_)
        #print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
    print(estimators)
elif grid == 4:
    df_m = df[df.sex == 'M']
    X = df_m.drop(['Target','sex_0','sex_1','sex'],axis=1)
    y = df_m.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=15)                      # number of folds
        clf.fit(X, y)
        print("Tuned Hyperparameters :", clf.best_params_)
        print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
        #clf.fit(X_5_train, y_5_train)
        #print("Tuned Hyperparameters :", clf.best_params_)
        #print("Accuracy :", clf.best_score_)
        # add the clf to the estimators list
        estimators.append((classifier.__class__.__name__, clf))
    print(estimators)
else:
    print('no grid selected')




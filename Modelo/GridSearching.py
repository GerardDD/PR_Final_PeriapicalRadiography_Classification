import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import KBinsDiscretizer
import category_encoders as ce
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
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

df_2 = pd.read_csv('df_rotated_unCens.csv')
df_2.drop('Unnamed: 0',axis=1,inplace=True)
df_2.rename(columns={'1000':'age', '1001':'sex'},inplace=True)
encoder = ce.BinaryEncoder()
df_2['sex_0'] = encoder.fit_transform(df_2['sex'])['sex_0']
df_2['sex_1'] = encoder.fit_transform(df_2['sex'])['sex_1']
kdisc = KBinsDiscretizer(n_bins=7, encode='ordinal')
df_2['age_bin'] = kdisc.fit_transform(df_2[['age']])
X_2 = df_2.drop(['Target','sex','age'],axis=1)
y_2 = df_2.Target

df_3 = pd.read_csv('df_rotated_unCens_vgg19.csv')
df_3.drop('Unnamed: 0',axis=1,inplace=True)
df_3.rename(columns={'1000':'age', '1001':'sex'},inplace=True)
encoder = ce.BinaryEncoder()
df_3['sex_0'] = encoder.fit_transform(df_3['sex'])['sex_0']
df_3['sex_1'] = encoder.fit_transform(df_3['sex'])['sex_1']
kdisc = KBinsDiscretizer(n_bins=7, encode='ordinal')
df_3['age_bin'] = kdisc.fit_transform(df_3[['age']])

df_4 = pd.read_csv('dataset_selected_as.csv')
df_4.drop('Unnamed: 0',axis=1,inplace=True)
df_4.rename(columns={'1000':'age', '1001':'sex'},inplace=True)
encoder = ce.BinaryEncoder()
df_4['sex_0'] = encoder.fit_transform(df_4['sex'])['sex_0']
df_4['sex_1'] = encoder.fit_transform(df_4['sex'])['sex_1']
kdisc = KBinsDiscretizer(n_bins=7, encode='ordinal')
df_4['age_bin'] = kdisc.fit_transform(df_4[['age']])

classifiers = [
    NuSVC(), 
    NearestCentroid(), 
    GaussianNB(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    AdaBoostClassifier()
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

xgboost_parameters = {
    'classifier__gamma': np.linspace(0,1,5),
    'classifier__booster': ['gbtree', 'dart'],
    'classifier__max_depth': [5,10,15],
    'classifier__learning_rate': [0.05,0.025]
}

adaboost_parameters = {
    'classifier__n_estimators': [50,150,100],
    'classifier__learning_rate': [0.05,0.025]
}

parameters = [
    NuSVC_parameters,
    NearestCentroid_parameters,
    GaussianNB_parameters,
    RandomForest_parameters,
    ExtraTree_parameters,
    xgboost_parameters,
    adaboost_parameters

]

estimators = []
# iterate through each classifier and use GridSearchCV
grid = int(input('choose grid \n'))
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
elif grid == 5:

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=15)                      # number of folds
        clf.fit(X_2, y_2)
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

elif grid == 6:
    X_2 = df_2.drop(['Target','sex','age','age_bin'],axis=1)
    y_2 = df_2.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=15)                      # number of folds
        clf.fit(X_2, y_2)
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
elif grid == 7:

    df_m2 = df_2[df_2.sex == 'M']
    X_2 = df_m2.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_2 = df_m2.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_2, y_2)
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
elif grid == 8:

    df_f2 = df_2[df_2.sex == 'F']
    X_2 = df_f2.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_2 = df_f2.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_2, y_2)
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
elif grid == 9:
    X_3 = df_3.drop(['Target','sex','age','age_bin'],axis=1)
    y_3 = df_3.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_3, y_3)
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
elif grid == 10:
    X_3 = df_3.drop(['Target','sex','age'],axis=1)
    y_3 = df_3.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        X_train, X_test, y_train, y_test = train_test_split(X_3, y_3,test_size=0.15,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_train, y_train)
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
elif grid == 11:
    df_f3 = df_3[df_3.sex == 'F']
    X_3 = df_f3.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_3 = df_f3.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_3, y_3)
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
elif grid == 12:
    df_f3 = df_3[df_3.sex == 'M']
    X_3 = df_f3.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_3 = df_f3.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=30)                      # number of folds
        clf.fit(X_3, y_3)
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
elif grid == 14:
    df_f4 = df_4[df_4.sex == 'F']
    X_4 = df_f4.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_4 = df_f4.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=20)                      # number of folds
        clf.fit(X_4, y_4)
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
elif grid == 15:
    #df_f4 = df_4[df_4.sex == 'F']
    X_4 = df_4.drop(['Target','sex_0','sex_1','sex','age'],axis=1)
    y_4 = df_4.Target

    estimators = []
# iterate through each classifier and use GridSearchCV
    for i, classifier in enumerate(classifiers):
        #X_train, X_test, y_train, y_test = train_test_split(X_2, y_2,test_size=0.2,shuffle=True)
        # create a Pipeline object
        print(f'I"m in cycle {i}')
        pipe = Pipeline(steps=[
            ('classifier', classifier)
        ])
        clf = GridSearchCV(pipe,              # model
                  param_grid = parameters[i], # hyperparameters
                  #scoring='accuracy',         # metric for scoring
                  cv=20)                      # number of folds
        clf.fit(X_4, y_4)
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




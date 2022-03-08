import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from lazypredict.Supervised import LazyClassifier

df = pd.read_csv('./dataset_final.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

X = df.drop('Target',axis=1)
y = df.Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)



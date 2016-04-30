import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

dataset = "/Users/dannyg/Dropbox/Datasets/SocialMediaBuzzDataset/Twitter/Absolute_labeling/Twitter-Absolute-Sigma-500.csv"

data = pd.read_csv(dataset, sep=',')

from sklearn.cross_validation import train_test_split

X = data.ix[: , :76] #features
Y = data.ix[: , 77] #buzz or no buzz

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

abst_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')

abst_clf.fit(x_train, y_train)

predictions = abst_clf.predict(x_test)

from sklearn.metrics import accuracy_score

print ('Accuracy Score: %s' % accuracy_score(y_test, predictions))
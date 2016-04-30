import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

dataset = "/Users/dannyg/Dropbox/Datasets/SocialMediaBuzzDataset/Twitter/Absolute_labeling/Twitter-Absolute-Sigma-500.csv"

data = pd.read_csv(dataset, sep=',')

from sklearn.cross_validation import train_test_split

X = data.ix[: , :76] #features
Y = data.ix[: , 77] #buzz or no buzz

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
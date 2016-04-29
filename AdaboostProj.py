import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

dataset = "/Users/dannyg/Dropbox/Datasets/SocialMediaBuzzDataset/Twitter/Absolute_labeling/Twitter-Absolute-Sigma-500.csv"

data = pd.read_csv(dataset, sep=',')

print (data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

plot_colours = 'br' #blue & red
plot_step = 0.02
class_names = 'AB'

plt.figure(figsize=(10,5))

#Plot the decision boundaries
plt.subplot(121)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = Y[0, :].min() - 1, Y[0, :].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = abst_clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.axis("tight")

print (xx, yy)

plt.show()
























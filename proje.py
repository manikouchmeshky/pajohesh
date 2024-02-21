import pandas as pd

data = pd.read_csv('diabetes.csv', names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
# print(data)
# skines = data[data['skin'] == 0]
# print(skines)
# print(len(skines))
# DS=data.describe()
# print(DS)
# print(DS.skin)

import matplotlib.pyplot as plt
# data['skin'].hist(bins=30, figsize=(5, 5))
# plt.show()
#
data['test'].hist(bins=30, figsize=(5, 5))
plt.title('test')
plt.show()
# skin_mean = data[data['skin'] != 0]['skin'].mean()
# data.replace({'skin': 0}, skin_mean, inplace=True)
# DS_clean=data.describe()
# print(DS_clean.skin)
# import matplotlib.pyplot as plt
# data['skin'].hist(bins=30, figsize=(5, 5))
# plt.show()

# دوباره دیتا را برای استفاده از sklearn میخوانیم
#
import numpy as np
# data = pd.read_csv('diabetes.csv', names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
data.replace({'test': 0, 'skin': 0}, np.nan, inplace=True)
# print(data)
# print(type(data))
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
data = imputer.fit_transform(data)
# print(data)
# plt.hist(data[:,3],bins=30)
# plt.show()
plt.hist(data[:,4],bins=30)
plt.title('test')
plt.grid()
plt.show()
# data=data[:,:-1]
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data)
#
#
print(data)
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# print(data)
#
# from sklearn.preprocessing import Normalizer
# norm = Normalizer()
# data = norm.fit_transform(data)
#
# #print(np.sum(data[0]**2))#
#
# print(data)

# STEP 3
x=data[:,0:8]
y=data[:,8]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y)

# from sklearn.tree import DecisionTreeRegressor
# tree = DecisionTreeRegressor().fit(X_train,y_train)
#


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

print("Training set accuracy: {:.3f}".format(dtree.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(dtree.score(X_test, y_test)))

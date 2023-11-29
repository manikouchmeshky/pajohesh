import pandas as pd
data = pd.read_csv('diabetes.csv', names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
print(data)
skines = data[data['skin'] == 0]
print(skines)
print(len(skines))
DS=data.describe()
#print(DS)
print(DS.skin)

skin_mean = data[data['skin'] != 0]['skin'].mean()
data.replace({'skin': 0}, skin_mean, inplace=True)
DS_clean=data.describe()
print(DS_clean.skin)

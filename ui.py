import pandas as pd
import xgboost as xgb
import tkinter as tk

def process_input():
    user_input = entry.get()
    print(f"You entered: {user_input}")

root = tk.Tk()
root.title("Input to Data Converter")

def women ():
    labels = ["قد", "وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست", "تعداد فرزند"]
    entries = []

    for i, item in enumerate(labels):
        label = tk.Label(root, text=item)
        label.grid(column=0, row=i)
        entry = tk.Entry(root)
        entry.grid(column=1, row=i)
        entries.append(entry)

    def process_all_inputs():
        l=[]
        for i, entry in enumerate(entries):
            user_input = entry.get()
            l.append(int(user_input))
        print(l)
        BMI=l[0]/(l[1]*l[1])
        feshar=l[2]
        ghand=l[3]
        sen=l[4]
        pelasma=l[5]
        verasat=l[6]
        zekhamat=l[7]
        farzand=l[8]
        xinput = [farzand, pelasma, feshar, zekhamat, ghand, BMI, verasat, sen]
        return xinput
    label = ["قد", "وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست","تعداد فرزند"]
    process_button = tk.Button(root, text="تایید", command=process_all_inputs)
    process_button.grid(column=0, row=len(label), columnspan=2)
def men ():
    # import graphviz as
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
    # data['test'].hist(bins=30, figsize=(5, 5))
    # plt.title('test')
    # plt.show()
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
    # plt.hist(data[:, 4], bins=30)
    # plt.title('test')
    # plt.grid()
    # plt.show()
    # data=data[:,:-1]
    #
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(data)
    #
    #
    # print(data)
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
    X = data[:, 0:8]
    y = data[:, 8]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=123)
    xgb_clf = xgb.XGBClassifier(random_state=123)
    xgb_clf.get_params()
    # from sklearn.tree import DecisionTreeRegressor
    # tree = DecisionTreeRegressor().fit(X_train,y_train)
    #

    # from sklearn.tree import DecisionTreeClassifier
    # dtree = DecisionTreeClassifier()
    # dtree = dtree.fit(X_train,y_train)
    xgb_clf.set_params(n_estimators=20)
    xgb_clf.fit(X_train, y_train)
    preds = xgb_clf.predict(X_test)
    # print("Training set accuracy: {:.3f}".format(dtree.score(X_train, y_train)))
    # print("Test set accuracy: {:.3f}".format(dtree.score(X_test, y_test)))
    accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
    # print("Baseline accuracy:", accuracy)
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (10.0, 8)
    xgb.plot_importance(xgb_clf)
    plt.show()


    labels = ["قد", "وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست"]
    entries = []

    for i, item in enumerate(labels):
        label = tk.Label(root, text=item)
        label.grid(column=0, row=i)
        entry = tk.Entry(root)
        entry.grid(column=1, row=i)
        entries.append(entry)

    def process_all_inputs():
        l=[]
        for i, entry in enumerate(entries):
            user_input = entry.get()
            l.append(int(user_input))
        BMI=l[0]/(l[1]*l[1])
        feshar=l[2]
        ghand=l[3]
        sen=l[4]
        pelasma=l[5]
        verasat=l[6]
        zekhamat=l[7]
        xinput = [0, pelasma, feshar, zekhamat, ghand, BMI, verasat, sen]
        preds = xgb_clf.predict(xinput)
        print('****'+ preds)
        # return xinput
    labels = ["قد","وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست"]
    process_button = tk.Button(root, text="تایید", command=process_all_inputs)
    process_button.grid(column=0, row=len(labels), columnspan=2)

labels = ["وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست"]
process_button = tk.Button(root, text="آقا", command=men)
process_button.grid(column=0, row=len(labels), columnspan=2)

label = ["وزن", "فشار خون", "قند خون", "سن", "پلسمای خون", "وراثت دیابت", "ضخامت پوست" , "تعداد فرزند"]
button = tk.Button(root, text="خانم", command=women)
button.grid(column=0, row=len(label), columnspan=2)

root.mainloop()



# matplotlib.rcParams['figure.figsize'] = (30.0, 20)
# xgb.plot_tree(xgb_clf, num_trees=0);

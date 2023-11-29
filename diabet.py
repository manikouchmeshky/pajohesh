# بارگذاری کتابخانه‌های مورد نیاز
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# بارگذاری مجموعه داده
df = pd.read_csv("diabetes.csv")

# جدا کردن ورودی‌ها و خروجی‌ها
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# تقسیم داده‌ها به دو بخش آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نرمال سازی داده‌ها
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# ساخت مدل شبکه عصبی
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation="relu", input_shape=(8,)))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# کامپایل مدل
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# آموزش مدل
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# پیشبینی بر روی داده‌های آزمون
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

# چاپ نتایج
print("دقت مدل:", accuracy_score(y_test, y_pred))
print("ماتریس درهم ریختگی:")
print(confusion_matrix(y_test, y_pred))
print("گزارش دسته‌بندی:")
print(classification_report(y_test, y_pred))

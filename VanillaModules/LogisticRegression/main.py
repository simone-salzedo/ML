
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from logireg import LogiReg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
SEED = 42
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['diagnosis'] = bc.target
dfX = df.iloc[:,:-1]   # Features - 30 columns
dfy = df['diagnosis']  # Label - last column
X = dfX.values
y = dfy.values
sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
logreg = LogiReg(random_state=SEED, n_features=dfX.shape[1])
logreg.fit(x=x_train, y=y_train)
y_pred = logreg.predict(x=x_test)
print(y_pred)
"""Plotting ConfusionMatrix"""
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
Accuracy = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] +cm[0, 1] + cm[1, 0]))*100
print(f"Accuracy: {Accuracy:.2f} %")
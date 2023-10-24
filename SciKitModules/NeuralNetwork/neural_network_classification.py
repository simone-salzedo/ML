import pandas as pd
import numpy as np
import sklearn.neural_network
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay


np.random.seed(42)

transfusion_df = pd.read_csv('../../dataSets/transfusion.csv')
print(transfusion_df.columns)

transfusion_df.rename(columns={"whether he/she donated blood in March 2007": "donatedblood"}, inplace=True)

transfusion_df['donateblood'] = transfusion_df['donatedblood'].replace('y', 1)


# shuffle to avoid group bias
index = transfusion_df.index
transfusion_df = transfusion_df.iloc[np.random.choice(index, len(index))] #iloc ci fa usare gli interi per selezionare una cella dell'array e non le etichette

X = transfusion_df.drop(['donateblood'], axis=1).values #contiene i dati di trasfusion senza la colonna donateblood
y_label = transfusion_df['donateblood'].values

train_index = round(len(X)*0.8)

X_train = X[:train_index]
y_train = y_label[:train_index]

X_test = X[train_index:]
y_test = y_label[train_index:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

param_grid = {
    'hidden_layer_sizes': [(10, 5), (20, 10)],
    'alpha': [0.00001, 0.0001],
    'solver': ['lbfgs', 'adam']
}

nn = sklearn.neural_network.MLPClassifier(max_iter=1000, random_state=42, learning_rate='constant', activation='logistic')

clf = GridSearchCV(nn,
                   param_grid=param_grid,
                   scoring='accuracy',
                   cv=5,
                   verbose=3)

clf.fit(X_train, y_train)
print('Best params found are: ', clf.best_params_)
print('Best Accuracy found is: ', clf.best_score_)

#Training nn with best parameters:

nn1 = sklearn.neural_network.MLPClassifier(max_iter=1000, random_state=42, learning_rate='constant', activation='logistic', hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'], alpha=clf.best_params_['alpha'], solver=clf.best_params_['solver'])
nn1.fit(X_train, y_train)
y_pred = nn1.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)) #this is the same as best score


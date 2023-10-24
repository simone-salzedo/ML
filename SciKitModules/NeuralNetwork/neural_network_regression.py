import pandas as pd
import numpy as np
import sklearn.neural_network
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, \
    make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../dataSets/houses.csv')
#print(list(df))
#print(df.head())

"""Preprocessing (Shuffling)"""
df = df.dropna(axis=1) #Dropping out missing Values on columns
df = df.sample(frac=1).reset_index(drop=True) #shuffling values and dropping old index
#print(list(df))
#print(df.head())
X = df[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = df['SalePrice'].values
#print(X)
#print(y)
"""Preprocessing (Training Set/Test set Division and z-score Normalization)"""
sc = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""Preparing Model"""
nn = sklearn.neural_network.MLPRegressor(max_iter=10000, random_state=123, learning_rate='constant', activation='relu')
"""Setting Up GridSearch to find optimal parameters"""
param_grid = {
    #'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #'solver': ['lbfgs', 'sgd'],
    #'hidden_layer_sizes': [(10, 5), (20, 10), (30, 15), (40, 20)]
    'alpha': [0.001],
    'solver': ['lbfgs'],
    'hidden_layer_sizes': [(4)]
}
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)
clf = GridSearchCV(nn,
                   param_grid=param_grid,
                   scoring=neg_mean_squared_error_scorer,
                   cv=5,
                   verbose=3)
clf.fit(X_train, y_train)
print('Best neg_MSE found: ', clf.best_score_)
print('With paramenters: ', clf.best_params_)

nn1 = sklearn.neural_network.MLPRegressor(max_iter=1000, random_state=123, learning_rate='constant', activation='relu',
                                          alpha=clf.best_params_['alpha'], solver=clf.best_params_['solver'],
                                          hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'])
nn1.fit(X_train, y_train)
y_preds = nn1.predict(X_test)
print(mean_squared_error(y_preds, y_test))


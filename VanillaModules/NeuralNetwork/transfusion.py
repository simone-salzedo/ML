import pandas as pd
import numpy as np

from neural_network_classification import NeuralNetwork

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

nn = NeuralNetwork(learning_rate=0.01, epochs=700, lmd=1, layers=[X_train.shape[1], 5, 5])

nn.fit(X_train, y_train)



nn.plot_loss()

preds = nn.predict(X_test)

print(nn.accuracy(y_test, preds[-1]))
y_pred = nn.predict(X_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from matplotlib import cm

plt.style.use(['ggplot'])

# read the dataset of houses prices
houses_1 = pd.read_csv('../../dataSets/houses.csv')
houses_1=houses_1.dropna(axis=1)
# print dataset stats
print(houses_1.describe())

# shuffling all samples to avoid group bias
houses = houses_1.sample(frac=1).reset_index(drop=True)
# select only some features, also you can try with other features (values return a numpy array)
x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values

# select target value (values return a numpy array)
y = houses['SalePrice'].values

# in order to perform hold-out splitting 80/20 identify max train index value
"""function round returns a floating point number that is a
rounded version of the specified number, with the specified number of decimals.
with the function len we counting length of the tuples and/or list."""

train_index = round(len(x) * 0.8)
# split training set in training and validation with hold-out 70/30 and get index of splitting
validation_index = round(train_index * 0.7)

# split dataset into training and test
X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
"""In pre-processing the data set before applying a machine learning algorithm the data can
be centered by subtracting the mean of the variable, and scaled by dividing by the standard deviation."""
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add bias column
"np.ones return a new array of given shape and type, X_train.shape[0] given a number of observation in a column"
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# split training set into training and validation
X_validation = X_train[validation_index:]
y_validation = y_train[validation_index:]

X_train = X_train[:validation_index]
y_train = y_train[:validation_index]

"define a K-fold CV function with K = 5, (example of 5 Fold cv without training)"

def k_fold_cv(X, k=5):
    for i in range(k):
        val_indices = list(range(round(i * len(X) / k), round((i + 1) * len(X) / k)))
        yield np.delete(X, val_indices, axis=0), X[val_indices]


# create a regressor with specific characteristics
".shape[1] given a number of elements in row"
linear = LinearRegression(n_features=X_train.shape[1], n_steps=1000, learning_rate=0.01, lmd=0)

# fit (try different strategies) your trained regressor
cost_history, cost_history_val, theta_history = linear.fit(X_train, y_train, X_validation, y_validation)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost/MSE:  {cost_history[-1]:.3f}''')
print(f'''Final validation cost/MSE:  {cost_history_val[-1]:.3f}''')

# plot loss curves
fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
c, = ax.plot(range(linear.n_steps), cost_history, 'b.')
cv, = ax.plot(range(linear.n_steps), cost_history_val, 'r+')
c.set_label('Train cost')
cv.set_label('Valid cost')
ax.legend()

plt.show()

# compute performance of your regressor model
res = linear.compute_performance(X_test, y_test)
print('\nperformance:\n')
print(''.join(['%s = %s\n' % (key, value) for (key, value) in res.items()]))

# plot cost history evolution with reference to thetas (it's a 3D plot, please choose only 2 different thetas!!)
fig = plt.figure()

ax = fig.gca()
first_dim = 0
second_dim = 1

# Make grid data.
A = np.linspace(np.min(theta_history[:, first_dim]), np.max(theta_history[:, first_dim]), 100)
B = np.linspace(np.min(theta_history[:, second_dim]), np.max(theta_history[:, second_dim]), 100)
A, B = np.meshgrid(A, B)
Z = linear.cost_grid(X_train, y_train, A, B, first_dim, second_dim)

# Plot the surface.
surf = ax.plot(A, B, Z)

ax.plot(theta_history[:, first_dim], theta_history[:, second_dim], cost_history, label='parametric curve')
plt.show()

# compute and plot learning curves
cost_history, cost_history_val = linear.learning_curves(X_train, y_train, X_validation, y_validation)

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('# Training samples')
c, = ax.plot(range(len(cost_history)), cost_history, color="blue")
cv, = ax.plot(range(len(cost_history_val)), cost_history_val, color="red")
ax.set_yscale('log')
c.set_label('Train cost')
cv.set_label('Valid cost')
ax.legend()

plt.show()

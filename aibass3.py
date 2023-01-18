import pandas as pd #for data manipulation and analysis 
from sklearn.svm import SVR 
'''
for support vector regression model with RBF kernel function to predict the target variable from the feature
using the training data set and evaluate the performance of the model on the testing data set to predict the target
variable from the features using the testing data set.
'''
import numpy as np #for scientific computing with Python 
from bayes_opt import BayesianOptimization 
'''for Bayesian optimization to find the hyperparameters that maximize the mean 5-fold cross-validation score 
'''
from sklearn.model_selection import train_test_split #for splitting the data into training and testing sets 
import matplotlib.pyplot as plt #for plotting the data 
from sklearn.model_selection import cross_val_score #for cross-validation to evaluate the performance of the model on the training data 

# load the data from the CSV file into a pandas data frame to manipulate and analyze the data
data = pd.read_csv('dataset.csv')

# select the first five columns as the features and the sixth column as the target using the iloc function of the pandas data frame
X = data.iloc[:, :5]
y = data.iloc[:, 5]
""" 
convert the data to numeric values and replace any non-numeric values with NaN values and then drop the
rows with NaN values from the data set (this is done because the data set might contain some non-numeric values)  
"""
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
""" 
split the data into training and testing sets, 70% for training and 30% for testing to train the model and
evaluate the performance of the model on the testing set 
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
""" 
define the function to be optimized, which is the mean 5-fold cross-validation score, which is being maximized. 
The function takes the hyperparameters to be optimized over as arguments. The hyperparameters are C, gamma, and 
epsilon for the SVR model with RBF kernel function. The function returns the mean 5-fold cross-validation score 
which is being maximized.The function is passed to the optimizer as an argument. The optimizer will find the 
hyperparameters that maximize the mean 5-fold cross-validation score. Cross-validation is used to evaluate the 
performance of the model on the training data. The model is trained on 4/5 of the training data and evaluated on
the remaining 1/5 of the training data. This is repeated 5 times, each time using a different 1/5 of the training 
data as the evaluation set. The mean of the 5 scores is the mean 5-fold cross-validation score. The model is trained 
on the entire training set and evaluated on the testing set to get the final score. The mean 5-fold cross-validation
score is used to select the best model and the final score is used to evaluate the performance of the best model on 
the testing set. 
"""
def svr_rbf(C, gamma, epsilon):
    val = cross_val_score(SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon), X_train, y_train, cv = 5) 
    return val.mean()
""" 
bounds for hyperparameters to be optimized over (C, gamma, epsilon) for the SVR model with RBF kernel function to find 
the hyperparameters that maximize the mean 5-fold cross-validation score 
"""
bds = {'C': (1, 100), 'gamma': (0.001,10),'epsilon': (0.001, 1)}
""" 
initialize the optimizer with the function to be optimized and the bounds for the hyperparameters to be optimized over to find 
the hyperparameters that maximize the mean 5-fold cross-validation score 
"""
optimizer = BayesianOptimization(f=svr_rbf,pbounds=bds)
""" 
define the parameters for the Gaussian process model used to fit the function to be optimized and the acquisition function 
used to find the next point to evaluate the function to be optimized at and the number of iterations to run the optimizer 
for and the number of initial points to evaluate the function to be optimized at and the exploration-exploitation tradeoff
parameter for the acquisition function and the alpha parameter for the acquisition function which is the variance of the
noise in the function to be optimized. The default value is 1e-5. The higher the value, the more likely the optimizer is 
to explore new values for the hyperparameters. The lower the value, the more likely the optimizer is to exploit the current
best value for the hyperparameters. 
""" 
optimizer.maximize(n_iter=100,init_points=4,acq='ei', xi=0.5,alpha=1e-5)

""" 
print the optimized hyperparameters and the value of the function to be optimized at the optimized hyperparameters which 
is the mean 5-fold cross-validation score.
"""

print('\n\nBayesian Opt. Done ! \nHyperparams tuned using ei acq fn to give best mean 5 fold cv score. \n\nBest kernel param: ', optimizer.max)

"""
use the optimized parameters to train the model on the training set, which is the SVR model with RBF kernel function with 
the optimized hyperparameters: C, gamma, and epsilon. 
"""
svr = SVR(kernel='rbf', C=optimizer.max['params']['C'], gamma=optimizer.max['params']['gamma'], epsilon=optimizer.max['params']['epsilon'])

# train the model on the training set to learn the relationship between the features and the target values of the training set
svr.fit(X_train, y_train)

# use the trained model to predict the target values of the testing set to evaluate the performance of the model on the testing set 
y_pred = svr.predict(X_test)
print("The actual values of the target are: ", y_test)
print("The predicted values of the target are: ", y_pred)
#find deviation of each value of y_pred from each value in y_test and print
print("The total value of the deviation of each value of y_pred from each value in y_test is: ", 26*np.mean(np.abs(y_pred-y_test)))
#predict target variable for x_train to evaluate the performance of the model on the training set
yt_pred = svr.predict(X_train)

"""
subplot comparing performance of model on both testing and  training sets to evaluate the performance of the model on the testing set 
and the training set 
"""
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) #sharey=True makes the y-axis of both plots the same
fig.suptitle('Comparing performance of model on testing and training sets')
ax1.set_title('Testing set')
ax1.set_xlabel('Lattice parameter From dataset') #label the x-axis
ax1.set_ylabel('Lattice parameter Predicted By Model') #label the y-axis
ax1.plot(y_test, y_pred, 'r^', y_test, y_test, 'k') # k is the color black and r^ is the color red and the marker ^ is a triangle
ax2.set_title('Training set')
ax2.set_xlabel('Lattice parameter From dataset') #label the x-axis
ax2.plot(y_train, yt_pred, 'r^', y_train, y_train, 'k') # k is the color black and r^ is the color red and the marker ^ is a triangle
plt.show() #show the plot

"""
calculate the mean squared error which is the average of the squared differences between the predicted values and the actual 
values of the testing set to evaluate the performance of the model on the testing set. 
"""
mse = np.mean((y_pred - y_test)**2) 
print('\n\n\nMean squared error for testing set: ', mse)
""" 
calculate the coefficient of determination (R^2) which is the proportion of the variance in the target values that is predictable 
from the features to evaluate the performance of the model on the testing set. 
"""
r2 = svr.score(X_test, y_test)
print('Coefficient of determination for testing set: ', r2)

"""
calculate the mean absolute error which is the average of the absolute differences between the predicted values and the actual 
values of the testing set to evaluate the performance of the model on the testing set.
"""
mae = np.mean(abs(y_pred - y_test))
print('Mean absolute error for testing set: ', mae,"\n\n")

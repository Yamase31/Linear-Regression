import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


## Read in data
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())



#Write linear regression class
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        # initailize important variables
        self._eta = eta
        self._n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        # for loop that gets an output

        for j in range(self._n_iter):
            predictions = self.net_input(X)

            # calculates the errors
            error = (y - predictions)
        
            # adjusts the weights
            self.w_[1] += self._eta * X.T.dot(error)
            self.w_[0] += self._eta * error.sum()
        
            # calculates mean squared error
            meanSquaredError = (error**2).sum() / 2.0
            
            # appends that cost to self.cost
            self.cost_.append(meanSquaredError)
        
        return self

    def net_input(self, X):
        # return the dot product of the weights with the input and add the bias
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # returns the models prediction
        return self.net_input(X)



#Choose variables to consider
X = df[['RM']].values
y = df['MEDV'].values

#Scale data and fit to the scaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

#Call the linear regression class and the fit function you wrote
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

#Graph the resulting squared error for every epoch
plt.plot(range(1, lr._n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

#Graph the resulting linear line for all data
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

plt.show()

print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])

#Use model to find price for a house with 5 rooms
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))

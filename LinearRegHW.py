import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

"""

Your homework is to complete the TODO with a linear regression model
from scikit learn. You can read up on this on your own here:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

You can see an example of this model being created and used here:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

Read both these online examples and complete the code below
hint: you only need 3 lines of code

"""



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

#Choose variables to consider
X = df[['RM']].values
y = df['MEDV'].values

#Scale data and fit to the scaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

#Call the linear regression model from scikit learn
slr = LinearRegression()

#Make sure you get y predictions to create your best fit line below
# TODO; Call your model slr and use X_std and y_std as your data.
slr.fit(X_std, y_std)
y_prediction = slr.predict(X_std)

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, y_prediction, color='red', lw=2)    
    return

lin_regplot(X_std, y_std, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


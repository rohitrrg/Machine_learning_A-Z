# multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable trapped
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting 3. Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# print(regressor)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# print(y_pred)
# print('')
# print(y_test)

# Building the optimal model using Backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt1 = X[:, [0, 1, 3, 4, 5]]
regressor_OLS1 = sm.OLS(endog = y, exog = X_opt1).fit()
print(regressor_OLS1.summary())

X_opt2 = X[:, [0, 3, 4, 5]]
regressor_OLS2 = sm.OLS(endog = y, exog = X_opt2).fit()
print(regressor_OLS2.summary())

X_opt3 = X[:, [0, 3, 5]]
regressor_OLS3 = sm.OLS(endog = y, exog = X_opt3).fit()
print(regressor_OLS3.summary())

X_opt4 = X[:, [0, 3]]
regressor_OLS4 = sm.OLS(endog = y, exog = X_opt4).fit()
print(regressor_OLS4.summary())
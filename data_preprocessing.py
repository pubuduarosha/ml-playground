import numpy as np
import matplotlib.pyplot as pt
import pandas as pd 

data = pd.read_csv('Data/mydata.csv')

X = data.iloc[:,:-1].values

Y = data.iloc[:,3].values

from sklearn.preprocessing import Imputer

imputerVariable = Imputer(missing_values = 'NaN',
                          strategy = 'mean',
                          axis = 0)

imputerVariable.fit(X[:,1:3])

X[:, 1:3] = imputerVariable.transform(X[:, 1:3])

from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with the 'mean' strategy for imputation
imputerVariable = SimpleImputer(strategy='mean')

# Fit the imputer on the data and transform the specified columns (1 and 2)
X[:, 1:3] = imputerVariable.fit_transform(X[:, 1:3])
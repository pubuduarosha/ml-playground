import numpy as np
import matplotlib.pyplot as pt
import pandas as pd 

data = pd.read_csv('Data/mydata.csv')

X = data.iloc[:,:-1].values

Y = data.iloc[:,3].values

from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with the 'mean' strategy for imputation
imputerVariable = SimpleImputer(strategy='mean')

# Fit the imputer on the data and transform the specified columns (1 and 2)
X[:, 1:3] = imputerVariable.fit_transform(X[:, 1:3])    

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a LabelEncoder for the first column
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Create a OneHotEncoder and specify the column to one-hot encode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


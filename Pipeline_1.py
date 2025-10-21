import numpy as np
import pandas as pd
import sklearn

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# These are your training samples along with their labels
data = pd.read_csv('health_insurance_train.csv')
# Here we separate target (y) and features (X)
X = data.drop('whrswk', axis=1)
y = data['whrswk']
# divide columns into numerical and non-numerical features
numerical_feats = ['experience', 'kidslt6', 'kids618', 'husby']
categorical_feats = ['hhi', 'whi', 'hhi2', 'education', 'race', 'hispanic', 'region']
# transformer for numbers
numerical_transf = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())])
# transformer for non-numbers
categorical_transf = Pipeline(steps=[('imputer', SimpleImputer(strategy="most_frequent")), ('encoder', OneHotEncoder())])
#
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transf, numerical_feats),
        ('cat', categorical_transf, categorical_feats)
    ]
)

preprocessor.fit(X)

X_transformed = preprocessor.transform(X)

print(X_transformed[0])
# You need to extract the features and the regression target. The regression target is 'whrswk'.
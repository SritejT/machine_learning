import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from CustomFiles.CustomRegressor import CustomRegressor

X_train = pd.read_csv('datasets/black_friday/train.csv')

y_train = X_train['Purchase']

del X_train['Purchase'], X_train['User_ID'], X_train['Product_ID'], X_train['Product_Category_2'], X_train['Product_Category_3']

one_hot = OneHotEncoder()

X_train_prepared = one_hot.fit_transform(X_train)

X_train_prepared = X_train_prepared.toarray()

cst_reg = CustomRegressor(regularisation='none', iterations=2001, eta=0.1)

cst_reg.fit(X_train_prepared, y_train)


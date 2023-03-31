from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def one_hot(x):

    new_x = []

    for i in range(0, len(x)):

        if x[i] == 0:

            new_x.append(1)
            new_x.append(0)

        else:

            new_x.append(0)
            new_x.append(1)

    return np.asarray(new_x).reshape(len(y_train), 2)


X_train = pd.read_csv('datasets/titanic/train.csv')

X_test = pd.read_csv('datasets/titanic/test.csv')

X_train["family_size"] = X_train["SibSp"] + X_train["Parch"]
X_test["family_size"] = X_test["SibSp"] + X_test["Parch"]


def titles(x):

    titles = []

    for i in x["Name"]:

        if '.' in i:

            for j in i.split():

                if j in ['Countess.', 'Mme.', 'Mrs.', 'Lady.', 'Dona.']:

                    titles.append('Mrs')

                elif j in ['Mr.', 'Don.', 'Major.', 'Capt.', 'Jonkheer.', 'Rev.', 'Col.', 'Sir.']:

                    titles.append('Mr')

                elif j in ['Mlle.', 'Ms.', 'Miss.']:

                    titles.append('Miss')

                elif j == 'Dr.':

                    titles.append('Dr')

                elif j == 'Master.':

                    titles.append('Master')

    return titles


X_train["Title"] = titles(X_train)
X_test["Title"] = titles(X_test)

y_train = X_train['Survived'].values

del X_train['Survived'], X_train['Name'], X_train['PassengerId'], X_train['Ticket'], X_train['Cabin'], X_train['SibSp'], X_train['Parch']

del X_test['Name'], X_test['PassengerId'], X_test['Ticket'], X_test['Cabin'], X_test['SibSp'], X_test['Parch']

X_train_num = X_train.drop("Sex", axis=1)
X_train_num = X_train_num.drop("Title", axis=1)
X_train_num = X_train_num.drop("Embarked", axis=1)
X_train_num = X_train_num.drop("Pclass", axis=1)

num_attribs = list(X_train_num)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

X_train["Embarked"].fillna("S", inplace=True)

cat_attribs = ["Sex", "Title", "Embarked", "Pclass"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

X_train_prepared = full_pipeline.fit_transform(X_train)

svm_clf = SVC()

svm_clf.fit(X_train_prepared, y_train)

X_test = full_pipeline.transform(X_test)

y_test = svm_clf.predict(X_test)

predicted_data = pd.DataFrame(data=y_test, columns=['Survived'], index=list(np.arange(892, 1310)))

predicted_data.index.name = "PassengerId"

predicted_data.to_csv(r'/Users/ravindratummuru/Desktop/sritejtummuru/machine_learning/datasets/titanic/predicted_data'
                      r'.csv')

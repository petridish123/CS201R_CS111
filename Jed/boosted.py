import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OHE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
xgb.set_config(verbosity=2)



# read in data
df = pd.read_csv("DataSets/pre_final_train.csv")

y = df["Final Exam Grade"]
X = df.drop("Final Exam Grade", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
label = X.columns
le = LE()
# le = OHE()
y_train = np.array(y_train)
# y_train = y_train.reshape(-1, 1)
label = le.fit_transform(y_train)
y_test_labels = le.transform(y_test)
# print(label)

dtrain = xgb.DMatrix(X_train, label = label, missing = np.nan)

dtest = xgb.DMatrix(X_test, label = y_test_labels, missing = np.nan)

param = {'max_depth': 16, 'eta': 0.3, 'objective': 'multi:softmax', 'num_class': len(le.classes_)}

num_round = 1000
evallist = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(param, dtrain, num_round, evallist,early_stopping_rounds=10)

ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

print(ypred)
print(accuracy_score(y_test_labels, ypred))

xgb.plot_importance(bst)




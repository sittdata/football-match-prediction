import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

d = pd.read_csv('football.csv')
d = d.drop_duplicates().dropna()

d['Result'] = d['Result'].map({'AwayWin': 0, 'Draw': 1, 'HomeWin': 2})

X = d.iloc[:, 0:6].to_numpy()
y = d.iloc[:, 6].to_numpy().astype(np.int64)

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

with open("xgboostmodel.sav", "wb") as f:
    pickle.dump(model, f)

print("Saved xgboostmodel.sav")

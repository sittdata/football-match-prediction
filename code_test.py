import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.dummy import DummyClassifier

d = pd.read_csv('football.csv')
d = d.drop_duplicates().dropna()

d['Result'] = d['Result'].map({'AwayWin': 0, 'Draw': 1, 'HomeWin': 2})

X = d.iloc[:, 0:6].to_numpy()
y = d.iloc[:, 6].to_numpy().astype(np.int64)

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

with open("xgboostmodel.sav", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Baseline Acc:   {accuracy_score(y_test, baseline.predict(X_test)):.4f}")
print(f"Test Log Loss:  {log_loss(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['AwayWin', 'Draw', 'HomeWin']))

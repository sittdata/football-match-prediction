import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

d = pd.read_csv('football.csv').dropna()
X_raw = d.iloc[:, 0:6].to_numpy()
scaler = MinMaxScaler().fit(X_raw)

with open('xgboostmodel.sav', 'rb') as f:
    model = pickle.load(f)


def predict_upcoming_match(home_team, away_team, home_xg, away_xg, home_xga, away_xga, handicap, odds_fav):
    row = np.array([[home_xg, away_xg, home_xga, away_xga, handicap, odds_fav]])
    row_scaled = scaler.transform(row)

    probs = model.predict_proba(row_scaled)[0]
    pred = model.predict(row_scaled)[0]

    labels = [f"{away_team} Win", "Draw", f"{home_team} Win"]

    print(f"\n{home_team} vs {away_team} (Handicap: {handicap}, Odds: {odds_fav})")
    print(f"  {away_team:<15} Win : {probs[0] * 100:.1f}%")
    print(f"  {'Draw':<15}     : {probs[1] * 100:.1f}%")
    print(f"  {home_team:<15} Win : {probs[2] * 100:.1f}%")
    print(f"  Prediction          : {labels[pred]}")


if __name__ == '__main__':
    predict_upcoming_match(
        home_team="Liverpool",
        away_team="Man United",
        home_xg=1.34,
        away_xg=1.17,
        home_xga=1.13,
        away_xga=1.77,
        handicap=-1.0,
        odds_fav=1.95
    )

    predict_upcoming_match(
        home_team="Arsenal",
        away_team="Chelsea",
        home_xg=1.75,
        away_xg=0.98,
        home_xga=1.37,
        away_xga=1.55,
        handicap=-0.5,
        odds_fav=1.90
    )

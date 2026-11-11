# Football Match Outcome Prediction (xG & XGBoost)

Match outcome prediction pipeline (Home Win / Draw / Away Win) built on 5,524 matches across the top 5 European football leagues using Expected Goals (xG) rolling form and market opening handicap odds.

## Dataset & Features

- **Source:** Understat (xG/xGA) and market opening lines (`raw_football.csv`, 5,524 matches).
- **Target:** `Result` (`AwayWin`, `Draw`, `HomeWin`).
- **Features (6):**
  - `Home_xG`, `Away_xG`: 5-match rolling average Expected Goals scored.
  - `Home_xGA`, `Away_xGA`: 5-match rolling average Expected Goals conceded.
  - `Handicap`: Opening Asian Handicap line.
  - `Odds_Fav`: Opening decimal odds for the favorite.

Rolling features are sorted chronologically and shifted by 1 (`shift(1)`) so predictions only use matches played prior to kickoff.

## Evaluation Results

Evaluated on an 80/20 split (`n = 1,105` test matches):

| Metric | XGBoost | Majority Baseline |
|---|---|---|
| Train Accuracy | 51.87% | — |
| Test Accuracy | 51.13% | 44.25% |
| Multi-class Log Loss | 1.0018 | 1.0687 (prior) |

## Files

- `code_cleaning.py` — Deduplication, missing value handling, and shifted 5-match rolling xG feature engineering. Outputs `football.csv`.
- `code_train.py` — MinMax scaling, 80/20 split, and XGBoost training. Saves `xgboostmodel.sav`.
- `code_test.py` — Evaluates test accuracy, log loss, baseline comparison, and classification report.
- `predict_future.py` — Sample inference script computing win/draw/loss probabilities for upcoming fixtures.

## Usage

```bash
pip install -r requirements.txt

python code_cleaning.py
python code_train.py
python code_test.py
python predict_future.py
```

## Sample Output (`predict_future.py`)

```text
Liverpool vs Man United (Handicap: -1.0, Odds: 1.95)
  Man United      Win : 19.3%
  Draw                : 24.9%
  Liverpool       Win : 55.8%
  Prediction          : Liverpool Win

Arsenal vs Chelsea (Handicap: -0.5, Odds: 1.9)
  Chelsea         Win : 28.0%
  Draw                : 26.3%
  Arsenal         Win : 45.7%
  Prediction          : Arsenal Win
```

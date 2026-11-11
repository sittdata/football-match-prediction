import numpy as np
import pandas as pd

df = pd.read_csv('raw_football.csv')
print(f"Raw rows: {len(df)}")

# deduplicate and filter finished matches
df = df.drop_duplicates(subset=['match_id'])
df = df.dropna(subset=['home_goals', 'away_goals'])

df['ah_open'] = df['ah_open'].fillna(0.0)
df['ah_fav_price'] = df['ah_fav_price'].fillna(1.90)

df['kickoff_at'] = pd.to_datetime(df['kickoff_at'], errors='coerce')
df = df.sort_values('kickoff_at').reset_index(drop=True)

# rolling 5-match xG stats shifted by 1 to avoid leakage
h_rows = pd.DataFrame({
    'idx': df.index,
    'team': df['home_team'],
    'date': df['kickoff_at'],
    'xg_f': df['home_xg'],
    'xg_a': df['away_xg'],
    'is_home': 1
})
a_rows = pd.DataFrame({
    'idx': df.index,
    'team': df['away_team'],
    'date': df['kickoff_at'],
    'xg_f': df['away_xg'],
    'xg_a': df['home_xg'],
    'is_home': 0
})

t_logs = pd.concat([h_rows, a_rows], ignore_index=True).sort_values(['team', 'date']).reset_index(drop=True)

grouped = t_logs.groupby('team')
t_logs['s_xg_f'] = grouped['xg_f'].shift(1)
t_logs['s_xg_a'] = grouped['xg_a'].shift(1)

roll = t_logs.groupby('team')[['s_xg_f', 's_xg_a']].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
t_logs['xg_f_avg'] = roll['s_xg_f']
t_logs['xg_a_avg'] = roll['s_xg_a']

h_feats = t_logs[t_logs['is_home'] == 1].set_index('idx')
a_feats = t_logs[t_logs['is_home'] == 0].set_index('idx')

df['Home_xG'] = h_feats['xg_f_avg'].fillna(1.5).round(2)
df['Away_xG'] = a_feats['xg_f_avg'].fillna(1.2).round(2)
df['Home_xGA'] = h_feats['xg_a_avg'].fillna(1.2).round(2)
df['Away_xGA'] = a_feats['xg_a_avg'].fillna(1.5).round(2)
df['Handicap'] = df['ah_open'].round(2)
df['Odds_Fav'] = df['ah_fav_price'].round(2)

df['Result'] = np.where(
    df['home_goals'] > df['away_goals'], 'HomeWin',
    np.where(df['home_goals'] == df['away_goals'], 'Draw', 'AwayWin')
)

cols = ['Home_xG', 'Away_xG', 'Home_xGA', 'Away_xGA', 'Handicap', 'Odds_Fav', 'Result']
clean_df = df[cols].dropna()

clean_df.to_csv('football.csv', index=False)
print(f"Saved football.csv ({len(clean_df)} rows)")

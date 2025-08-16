import pandas as pd
import nfl_data_py as nfl

print("📥 Loading QB data for 2023...")
qb_data = nfl.import_weekly_data([2023])
qb_data = qb_data[qb_data['position'] == 'QB']
qb_data = qb_data.sort_values(['player_name', 'week'])

for window in [3, 5]:
    qb_data[f'fantasy_points_{window}game_avg'] = (
        qb_data.groupby('player_name')['fantasy_points']
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

qb_data['fantasy_points_trend'] = qb_data['fantasy_points_3game_avg'] / (qb_data['fantasy_points_5game_avg'] + 0.1)
qb_data['last_fantasy_points'] = qb_data.groupby('player_name')['fantasy_points'].shift(1)

def show_sample_projection(df, player):
    player_df = df[df['player_name'].str.contains(player, case=False)]
    if player_df.empty:
        print(f"❌ No data found for {player}")
        return
    latest = player_df.sort_values('week', ascending=False).iloc[0]
    print(f"\n📊 Latest Projection for {latest['player_name']}:")
    print(f"  Week: {latest['week']}")
    print(f"  3-Game Avg: {latest['fantasy_points_3game_avg']:.2f}")
    print(f"  5-Game Avg: {latest['fantasy_points_5game_avg']:.2f}")
    print(f"  Trend: {latest['fantasy_points_trend']:.2f}")
    print(f"  Last Actual: {latest['last_fantasy_points']:.2f}")

show_sample_projection(qb_data, "J.Allen")

qb_data.to_csv("data/processed/qb_rolling_projection.csv", index=False)
print("\n✅ QB projection saved to: data/processed/qb_rolling_projection.csv")
print("✅ QB projection completed.")
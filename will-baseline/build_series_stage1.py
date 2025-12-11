import pandas as pd

CSV_PATH = "data/statcast_2024.csv"


def load_pitch_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_games_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"game_pk", "game_date", "home_team", "away_team"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    games = (
        df.groupby("game_pk")[["game_date", "home_team", "away_team"]]
        .first()
        .reset_index()
    )

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games[games["game_date"] < "2024-10-01"]
    games = games.sort_values(["home_team", "game_date"]).reset_index(drop=True)
    print("=" * 60)
    print("---- Games table head -----")
    print(games.head(20))
    # print("---- Unique teams ----")
    # print(games.home_team.value_counts())
    return games


def assign_series_ids(games: pd.DataFrame) -> pd.DataFrame:
    series_ids = []
    current_series = 0
    prev_home = None
    prev_away = None



    for _, row in games.iterrows():
        home, away = row["home_team"], row["away_team"]
        if prev_home is not None and (home != prev_home or away != prev_away):
            current_series += 1
        series_ids.append(current_series)
        prev_home, prev_away = home, away

    games_with_series = games.copy()
    games_with_series["series_id"] = series_ids
    return games_with_series


if __name__ == "__main__":
    df = load_pitch_data(CSV_PATH)
    games = build_games_table(df)
    games = assign_series_ids(games)

    games_per_series = games.groupby("series_id").size()

    print("Total series:", games["series_id"].nunique())
    print("Games per series distribution:")
    print(games_per_series.value_counts().sort_index())
    print("Games head with series_id:")
    print(games.head())

    # six_game_series = games_per_series[games_per_series == 6]
    # print("Series with 6 games:")
    # print(six_game_series)
    # for sid in six_game_series.index:
    #     print(f"\nSeries ID {sid}:")
    #     print(games[games["series_id"] == sid])
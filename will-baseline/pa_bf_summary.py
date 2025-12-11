import pandas as pd

CSV_PATH = "data/statcast_2024.csv"
SUMMARY_PATH = "data/player_pa_bf_summary.csv"


def load_pitch_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def compute_pa_and_bf(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"game_pk", "at_bat_number", "batter", "pitcher"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    batter_pas = df[["game_pk", "at_bat_number", "batter"]].drop_duplicates()
    batter_counts = (batter_pas.groupby("batter").size().rename_axis("player_id").reset_index(name="total_pa"))

    pitcher_pas = df[["game_pk", "at_bat_number", "pitcher"]].drop_duplicates()
    pitcher_counts = (pitcher_pas.groupby("pitcher").size().rename_axis("player_id").reset_index(name="total_bf"))

    combined = batter_counts.merge(pitcher_counts, on="player_id", how="outer").fillna(0)
    combined[["total_pa", "total_bf"]] = combined[["total_pa", "total_bf"]].astype(int)

    def assign_role(row: pd.Series) -> str:
        if row["total_pa"] >= row["total_bf"]:
            return "batter"
        return "pitcher"

    combined["role"] = combined.apply(assign_role, axis=1)
    combined = combined[["player_id", "role", "total_pa", "total_bf"]]
    return combined


def build_core_player_universe(df: pd.DataFrame) -> pd.DataFrame:
    team_cols = {"home_team", "away_team", "inning_topbot"}    
    missing_team = team_cols - set(df.columns)
    if missing_team:
        raise ValueError(f"Missing required team columns: {', '.join(sorted(missing_team))}")

    pa_bf = compute_pa_and_bf(df)

    inning_half = df["inning_topbot"].astype(str).str.lower()
    bat_team_col = pd.Series(pd.NA, index=df.index)
    fld_team_col = pd.Series(pd.NA, index=df.index)
    bat_team_col = bat_team_col.mask(inning_half.str.startswith("top"), df["away_team"])
    bat_team_col = bat_team_col.mask(~inning_half.str.startswith("top"), df["home_team"])
    fld_team_col = fld_team_col.mask(inning_half.str.startswith("top"), df["home_team"])
    fld_team_col = fld_team_col.mask(~inning_half.str.startswith("top"), df["away_team"])

    df_with_teams = df.assign(bat_team=bat_team_col, fld_team=fld_team_col)

    def most_frequent(series: pd.Series):
        counts = series.dropna().value_counts()
        return counts.idxmax() if not counts.empty else pd.NA

    batter_teams = (df_with_teams.groupby("batter")["bat_team"].agg(most_frequent).rename_axis("player_id").reset_index(name="team"))
    pitcher_teams = (df_with_teams.groupby("pitcher")["fld_team"].agg(most_frequent).rename_axis("player_id").reset_index(name="team"))

    team_map = (
        pd.concat([batter_teams, pitcher_teams], ignore_index=True)
        .groupby("player_id")["team"]
        .agg(most_frequent)
        .reset_index()
    )

    core_players = pa_bf[(pa_bf["total_pa"] >= 150) | (pa_bf["total_bf"] >= 150)]
    core_players = core_players.merge(team_map, on="player_id", how="left")
    return core_players[["player_id", "role", "total_pa", "total_bf", "team"]]


if __name__ == "__main__":
    data = load_pitch_data(CSV_PATH)
    summary = compute_pa_and_bf(data)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved summary to {SUMMARY_PATH}")

    core_players = build_core_player_universe(data)

    num_core_batters = (core_players["role"] == "batter").sum()
    num_core_pitchers = (core_players["role"] == "pitcher").sum()

    batter_team_counts = (
        core_players.loc[core_players["role"] == "batter"]
        .dropna(subset=["team"])
        .groupby("team")
        .size()
    )
    pitcher_team_counts = (
        core_players.loc[core_players["role"] == "pitcher"]
        .dropna(subset=["team"])
        .groupby("team")
        .size()
    )

    avg_batters_per_team = batter_team_counts.mean() if not batter_team_counts.empty else 0
    avg_pitchers_per_team = pitcher_team_counts.mean() if not pitcher_team_counts.empty else 0
    min_batters_per_team = batter_team_counts.min() if not batter_team_counts.empty else 0
    min_pitchers_per_team = pitcher_team_counts.min() if not pitcher_team_counts.empty else 0

    print("Summary preview:")
    print(summary.head())
    print(f"Total players: {len(summary)}")
    print(f"Max PA: {summary['total_pa'].max()}")
    print(f"Max BF: {summary['total_bf'].max()}")

    print("Total core players:", len(core_players))
    print("Core batters:", num_core_batters)
    print("Core pitchers:", num_core_pitchers)
    print("Avg batters per team:", avg_batters_per_team)
    print("Avg pitchers per team:", avg_pitchers_per_team)
    print("Min batters on any team:", min_batters_per_team)
    print("Min pitchers on any team:", min_pitchers_per_team)

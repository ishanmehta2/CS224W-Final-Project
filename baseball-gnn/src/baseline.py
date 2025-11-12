from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pybaseball import playerid_reverse_lookup, statcast

DOWNLOAD = False
COLUMNS = ["game_date", "game_pk", "batter", "pitcher", "stand", "p_throws", "inning", "outs_when_up", "bat_score", "fld_score", "events", "woba_value"]


def main():
    datadir = Path(__file__).resolve().parents[1] / "data"
    data = statcast("2023-04-01", "2023-04-30") if DOWNLOAD else pd.read_parquet(datadir / "statcast_2023_small.parquet")
    pa = data[COLUMNS].dropna(subset=["events", "woba_value"]).copy()
    for col, label in [("batter", "batter_name"), ("pitcher", "pitcher_name")]:
        tbl = playerid_reverse_lookup(pa[col].unique().tolist(), key_type="mlbam")
        pa[label] = pa[col].map(dict(zip(tbl["mlbam"], tbl["name_first"] + " " + tbl["name_last"])))
    pa["batter_idx"] = pd.factorize(pa["batter"])[0]
    pa["pitcher_idx"] = pd.factorize(pa["pitcher"])[0]
    stand = pd.get_dummies(pa["stand"]).reindex(columns=["L", "R", "S"], fill_value=0)
    throws = pd.get_dummies(pa["p_throws"]).reindex(columns=["L", "R"], fill_value=0)
    edge_attr = np.concatenate(
        [
            stand.to_numpy(),
            throws.to_numpy(),
            (pa["inning"] / 9).to_numpy()[:, None],
            pa["outs_when_up"].to_numpy()[:, None],
            (pa["bat_score"] - pa["fld_score"]).to_numpy()[:, None],
        ],
        axis=1,
    )
    graph = {
        "edge_index": torch.tensor([pa["batter_idx"].to_numpy(), pa["pitcher_idx"].to_numpy()], dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "y": torch.tensor(pa["woba_value"].to_numpy(), dtype=torch.float32),
        "num_batters": int(pa["batter_idx"].nunique()),
        "num_pitchers": int(pa["pitcher_idx"].nunique()),
        "num_edges": len(pa),
    }
    datadir.mkdir(parents=True, exist_ok=True)
    pa.to_parquet(datadir / "pa.parquet", index=False) 
    torch.save(graph, datadir / "graph.pt")
    print(
        f"batters={graph['num_batters']} pitchers={graph['num_pitchers']} "
        f"edges={graph['num_edges']} edge_attr_shape={graph['edge_attr'].shape}"
    )


if __name__ == "__main__":
    main()

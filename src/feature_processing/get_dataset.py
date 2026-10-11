import os
import sys
import glob
import pandas as pd
import fastf1 as f1
from fastf1 import get_session as gs
from fastf1.exceptions import RateLimitExceededError
from datetime import datetime, timedelta

from feature_functions import generate_features_per_race

RATE_LIMITED = 75   # must match RATE_LIMITED in the Makefile
SEASONS = range(2022, 2027)

os.makedirs("cache", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
f1.Cache.enable_cache("cache")

cutoff = datetime.today() - timedelta(days=4)


def checkpoint_path(season, round_number):
    return f"checkpoints/{season}_{round_number:02d}.csv"


def get_race_target(season, round_number):
    """Top-10 label per driver for one race. Returns None if results aren't available."""
    race = gs(season, round_number, "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)

    results = race.results[["DriverNumber", "Position"]].copy()
    if results.empty or results["Position"].isna().all():
        return None

    results["top10"] = (results["Position"] <= 10).astype(int)
    results["season"] = season
    results["race"] = race.event.OfficialEventName
    results = results.rename(columns={"DriverNumber": "driver"})
    results["driver"] = results["driver"].astype(str)
    return results[["season", "race", "driver", "top10"]]


def build_gp_dataset(season, round_number):
    """Features + target for one Grand Prix, with checkpointing."""
    path = checkpoint_path(season, round_number)
    if os.path.exists(path):
        print(f"[skip] {season} round {round_number} already downloaded")
        return

    print(f"[get ] {season} round {round_number}")

    target = get_race_target(season, round_number)
    if target is None:
        print(f"[warn] no race results for {season} round {round_number}, not checkpointing")
        return

    feats = generate_features_per_race(round_number, season)
    if feats.empty:
        print(f"[warn] no features for {season} round {round_number}, not checkpointing")
        return

    feats["driver"] = feats["driver"].astype(str)
    df = feats.merge(target, how="left", on=["season", "race", "driver"])
    df.to_csv(path, index=False)   # written only once the whole GP succeeded
    print(f"[save] {path} ({len(df)} rows)")


def main():
    rate_limited = False
    try:
        for season in SEASONS:
            schedule = f1.get_event_schedule(season)
            valid = (schedule.RoundNumber > 0) & (schedule.Session5DateUtc <= cutoff)
            for round_number in schedule.loc[valid, "RoundNumber"]:
                build_gp_dataset(season, int(round_number))
    except RateLimitExceededError:
        rate_limited = True
        print("\nRate limit hit. Progress is saved in checkpoints/.")

    # Always combine whatever has been downloaded so far
    files = sorted(glob.glob("checkpoints/*.csv"))
    if files:
        all_data = pd.concat(
            (pd.read_csv(f, dtype={"driver": str}) for f in files),
            ignore_index=True,
        )
        all_data.to_csv("all_data.csv", index=False)
        print(f"all_data.csv written from {len(files)} Grand Prix checkpoints")

    return RATE_LIMITED if rate_limited else 0


if __name__ == "__main__":
    sys.exit(main())
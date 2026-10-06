import pandas as pd
import fastf1 as f1
from fastf1 import get_session as gs
from datetime import datetime

from feature_functions import *

today = datetime.today()
SEASON = 2026
SESSIONS = ["FP1", "FP2", "FP3"]

TARGET_COLUMNS = ["driver", "race", "top10"]


# get target

all_events = f1.get_event_schedule(SEASON)
valid_events_condition = (all_events.RoundNumber > 0) & (all_events.Session1DateUtc <= today)
total_races = all_events[["RoundNumber", "OfficialEventName"]][valid_events_condition]
total_races.reset_index(drop=True, inplace=True)
all_race_results = []

for row in total_races.itertuples():
    race = gs(SEASON, row.RoundNumber, 'R')
    race.load()
    result = race.results[["DriverNumber", "Position"]]
    result["officialName"] = row.OfficialEventName
    all_race_results.append(result)
    
    
all_race_results = pd.concat(all_race_results, ignore_index=True)
all_race_results["top10"] = all_race_results["Position"] <= 10
all_race_results["top10"] = all_race_results["top10"].astype(int)
column_remame = {"DriverNumber" : "driver", "officialName" : "race"}
all_race_results.rename(columns = column_remame, inplace = True)

target_df = all_race_results[TARGET_COLUMNS]


# get features

all_valid_races = list(target_df.race.unique())

all_races_features_dfs = [generate_features_per_race(race) for race in all_valid_races]

all_races_features = pd.concat(all_races_features_dfs)

final_dataset = pd.merge(all_races_features, target_df, how = 'left', on = ["driver", "race"])

final_dataset.to_csv("final_dataset.csv")

print("final dataset written to final_dataset.csv")

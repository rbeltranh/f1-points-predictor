import numpy as np
import pandas as pd
from fastf1 import get_session as gs
SEASON = 2026
SESSIONS = ["FP1", "FP2", "FP3"]

def curvature(x, y):
    '''Takes the x and y coordinates and finds the curvarture per point, asuming the curve is 2-Dimensional'''
    x_t = np.gradient(x) # x_velocity
    y_t = np.gradient(y) # y_velocity
    speed = np.sqrt(x_t**2 + y_t**2)
    x_tt = np.gradient(x_t)
    y_tt = np.gradient(y_t)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = (x_tt * y_t - x_t * y_tt)/(speed**3)
    return  np.nan_to_num(abs(kappa))  


def get_diff_geom_feats(telemetry_df):
    telemetry_df["curvature"] = curvature(telemetry_df.X, telemetry_df.Y)
    speed_curvature_corr = telemetry_df["curvature"].corr(telemetry_df["Speed"])
    mean_curvature = telemetry_df["curvature"].mean()
    max_curvature = telemetry_df["curvature"].max()
    
    return speed_curvature_corr, mean_curvature, max_curvature

def get_driver_feats(driver, sesh, global_feats, season = SEASON):
    
    driver_laps = sesh.laps.pick_drivers(driver)
    if driver_laps.empty:
        print(f"Driver {driver} has not fp1 info, returning None")
        return None
    driver_fl = driver_laps.pick_fastest()
    try:
        driver_fl_tele = driver_fl.telemetry
    except (AttributeError, KeyError):
        print(f"Driver {driver} has not telemetry info, returning None")
        return None
    
    driver_feats = {}
    
    # Context Features
    driver_feats["race"] = sesh.event.OfficialEventName
    driver_feats["season"] = season
    driver_feats["driver"] = driver
    driver_feats["team"] = driver_fl.Team
    
    # Time features
    driver_feats["lap_time"] = driver_fl.Time.total_seconds()
    driver_feats["sector_1_time"] = driver_fl.Sector1Time.total_seconds()
    driver_feats["sector_2_time"] = driver_fl.Sector2Time.total_seconds()
    driver_feats["sector_3_time"] = driver_fl.Sector3Time.total_seconds()
    
    # Tyre features
    driver_feats["tyre_compound"] = driver_fl.Compound
    driver_feats["tyre_age"] = driver_fl.TyreLife
    
    # speed features
    
    driver_feats["mean_speed"] = driver_fl_tele.Speed.mean()
    driver_feats["max_speed"] = driver_fl_tele.Speed.max()
    driver_feats["speed_std"] = driver_fl_tele.Speed.std()
    
    # rpm features
    
    driver_feats["mean_rpm"] = driver_fl_tele.RPM.mean()
    driver_feats["max_rpm"] = driver_fl_tele.RPM.max()
    driver_feats["rpm_std"] = driver_fl_tele.RPM.std()
    
    # Throtle and gear features
    
    driver_feats["mean_throttle"] = driver_fl_tele.Throttle.mean()
    driver_feats["mean_gear"] = driver_fl_tele.nGear.mean()
    
    
    # Differential geometry features
    
    
    speed_curvature_corr, mean_curvature, max_curvature = get_diff_geom_feats(driver_fl_tele)
    
    driver_feats["speed_curvature_corr"] = speed_curvature_corr
    driver_feats["mean_curvature"] = mean_curvature
    driver_feats["max_curvature"] = max_curvature
    
    driver_feats["rpm_speed_corr"] = driver_fl_tele["RPM"].corr(driver_fl_tele["Speed"])
    driver_feats["throttle_speed_corr"] = driver_fl_tele["Throttle"].corr(driver_fl_tele["Speed"])
     
    return driver_feats | global_feats


def get_all_drivers_feats(sesh, global_feats, season):
    all_drivers_feats = []
    all_drivers = list(sesh.laps.DriverNumber.unique())
    for driver in all_drivers:
        print(f"Getting {sesh.name} features for driver {driver}")
        all_drivers_feats.append(get_driver_feats(driver, sesh, global_feats, season))
        
    return all_drivers_feats

def get_global_features(sesh):
    global_feats = {}
    fastest_lap = sesh.laps.pick_fastest().Time.total_seconds()
    global_feats["fastest_lap"] = fastest_lap
    # get weather info
    weather_data = sesh.weather_data
    ## Air temp
    global_feats["air_temp_mean"] = weather_data.AirTemp.mean()
    global_feats["air_temp_max"] = weather_data.AirTemp.max()
    global_feats["air_temp_min"] = weather_data.AirTemp.min()
    
    ## TrackTemp
    global_feats["track_temp_mean"] = weather_data.TrackTemp.mean()
    global_feats["track_temp_max"] = weather_data.TrackTemp.max()
    global_feats["track_temp_min"] = weather_data.TrackTemp.min()
    
    return global_feats


def generate_features_per_session(race: str, s: str, season = SEASON) -> pd.DataFrame:
    "Generates features per session given a specific race weekend and session"
    fp_sesh = gs(season, race, s)
    fp_sesh.load()
    global_feats = get_global_features(fp_sesh)
    all_drivers_feats = get_all_drivers_feats(fp_sesh, global_feats, season)
    clean_feats = [feat for feat in all_drivers_feats if feat]
    clean_feats_df = pd.DataFrame(clean_feats)
    clean_feats_df["session"] = s
    
    return clean_feats_df


def generate_features_per_race(race: str, season:str) -> pd.DataFrame:
    "Generate features for all sessions of a given race weekend"
    per_race_feats = []
    for s in SESSIONS:
        try:
            per_race_feats.append(generate_features_per_session(race, s, season))
        except ValueError as e:
            print(f"Skipping {s} for {race}: {e}")
            continue

    if not per_race_feats:
        return pd.DataFrame()
    
    return pd.concat(per_race_feats, ignore_index=True)
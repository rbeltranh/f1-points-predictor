"""Unit tests for the F1 feature-generation module.

Setup notes:
- Change the import below if your module is not called `features.py`.
- Make sure pytest can import it, e.g. in pytest.ini / pyproject.toml:
      [tool.pytest.ini_options]
      pythonpath = ["src/feature_processing"]
- No network or FastF1 cache is needed: every FastF1 object is mocked.
"""
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

import src.feature_processing.feature_functions as feat 


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
 
@pytest.fixture
def telemetry_df():
    """Small, deterministic telemetry frame for one lap (a circular track)."""
    n = 60
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "X": 100 * np.cos(t),
            "Y": 100 * np.sin(t),
            "Speed": rng.uniform(80, 320, n),
            "RPM": rng.uniform(8000, 12000, n),
            "Throttle": rng.uniform(0, 100, n),
            "nGear": rng.integers(1, 9, n).astype(float),
        }
    )
 
 
@pytest.fixture
def fastest_lap(telemetry_df):
    """Mock of the lap returned by `laps.pick_drivers(...).pick_fastest()`."""
    lap = MagicMock()
    lap.Team = "Ferrari"
    lap.LapTime = pd.Timedelta(seconds=90.5)
    lap.Sector1Time = pd.Timedelta(seconds=28.1)
    lap.Sector2Time = pd.Timedelta(seconds=35.2)
    lap.Sector3Time = pd.Timedelta(seconds=27.2)
    lap.Compound = "SOFT"
    lap.TyreLife = 3.0
    lap.telemetry = telemetry_df
    return lap
 
 
@pytest.fixture
def session(fastest_lap):
    """Mock FastF1 session whose driver lookup returns `fastest_lap`."""
    sesh = MagicMock()
    sesh.name = "Practice 1"
    sesh.event.OfficialEventName = "FORMULA 1 TEST GRAND PRIX 2026"
    driver_laps = sesh.laps.pick_drivers.return_value
    driver_laps.empty = False
    driver_laps.pick_fastest.return_value = fastest_lap
    return sesh
 
 
GLOBAL_FEATS = {"fastest_lap": 88.0, "air_temp_mean": 25.0, "track_temp_mean": 38.0}
 
 
# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------
 
class TestCurvature:
    def test_circle_has_curvature_one_over_radius(self):
        radius = 10.0
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        kappa = feat.curvature(radius * np.cos(t), radius * np.sin(t))
 
        # Skip the edges: np.gradient uses one-sided differences there.
        assert kappa[2:-2] == pytest.approx(1 / radius, rel=1e-2)
 
    def test_straight_line_has_zero_curvature(self):
        x = np.arange(10, dtype=float)
        kappa = feat.curvature(x, 2 * x)
        assert np.allclose(kappa, 0.0)
 
    def test_stationary_points_do_not_produce_nan_or_warnings(self, recwarn):
        # speed == 0 -> division by zero, which must be mapped to 0
        kappa = feat.curvature(np.ones(5), np.ones(5))
 
        assert np.all(np.isfinite(kappa))
        assert np.allclose(kappa, 0.0)
        assert len(recwarn) == 0
 
    def test_output_is_non_negative_and_same_length(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=50).cumsum(), rng.normal(size=50).cumsum()
        kappa = feat.curvature(x, y)
 
        assert kappa.shape == (50,)
        assert (kappa >= 0).all()
 
 
# ---------------------------------------------------------------------------
# get_diff_geom_feats
# ---------------------------------------------------------------------------
 
class TestGetDiffGeomFeats:
    def test_returns_corr_mean_and_max_from_curvature(self, monkeypatch):
        df = pd.DataFrame({"X": range(4), "Y": range(4), "Speed": [10.0, 20.0, 30.0, 40.0]})
        monkeypatch.setattr(feat, "curvature", lambda x, y: np.array([1.0, 2.0, 3.0, 4.0]))
 
        corr, mean_k, max_k = feat.get_diff_geom_feats(df)
 
        assert corr == pytest.approx(1.0)
        assert mean_k == pytest.approx(2.5)
        assert max_k == pytest.approx(4.0)
 
    def test_adds_curvature_column_to_input_frame(self, telemetry_df):
        assert "curvature" not in telemetry_df.columns
        feat.get_diff_geom_feats(telemetry_df)
        assert "curvature" in telemetry_df.columns
 
    def test_curvature_is_passed_the_x_and_y_columns(self, monkeypatch, telemetry_df):
        spy = MagicMock(return_value=np.zeros(len(telemetry_df)))
        monkeypatch.setattr(feat, "curvature", spy)
 
        feat.get_diff_geom_feats(telemetry_df)
 
        args = spy.call_args.args
        pd.testing.assert_series_equal(args[0], telemetry_df["X"])
        pd.testing.assert_series_equal(args[1], telemetry_df["Y"])
 
    def test_circular_track_mean_curvature(self):
        radius = 50.0
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        df = pd.DataFrame(
            {"X": radius * np.cos(t), "Y": radius * np.sin(t), "Speed": np.linspace(100, 300, 500)}
        )
        _, mean_k, _ = feat.get_diff_geom_feats(df)
 
        # Edge points are slightly off, so use a loose tolerance
        assert mean_k == pytest.approx(1 / radius, rel=0.05)
 
    def test_constant_speed_gives_nan_correlation(self):
        # Worth knowing: a constant Speed column yields NaN correlation,
        # which is one source of NaNs in the final dataset.
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        df = pd.DataFrame({"X": np.cos(t), "Y": np.sin(t), "Speed": np.full(50, 200.0)})
 
        corr, _, _ = feat.get_diff_geom_feats(df)
 
        assert np.isnan(corr)
 
 
# ---------------------------------------------------------------------------
# get_driver_feats
# ---------------------------------------------------------------------------
 
EXPECTED_DRIVER_KEYS = {
    "race", "season", "driver", "team",
    "lap_time", "sector_1_time", "sector_2_time", "sector_3_time",
    "tyre_compound", "tyre_age",
    "mean_speed", "max_speed", "speed_std",
    "mean_rpm", "max_rpm", "rpm_std",
    "mean_throttle", "mean_gear",
    "speed_curvature_corr", "mean_curvature", "max_curvature",
    "rpm_speed_corr", "throttle_speed_corr",
}
 
 
class TestGetDriverFeats:
    def test_returns_all_expected_keys_including_global_feats(self, session):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
 
        assert set(result) == EXPECTED_DRIVER_KEYS | set(GLOBAL_FEATS)
 
    def test_context_features(self, session):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS, season=2025)
 
        assert result["race"] == "FORMULA 1 TEST GRAND PRIX 2026"
        assert result["season"] == 2025
        assert result["driver"] == "16"
        assert result["team"] == "Ferrari"
 
    def test_season_defaults_to_module_constant(self, session):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
        assert result["season"] == feat.SEASON
 
    def test_time_and_tyre_features(self, session):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
 
        assert result["lap_time"] == pytest.approx(90.5)
        assert result["sector_1_time"] == pytest.approx(28.1)
        assert result["sector_2_time"] == pytest.approx(35.2)
        assert result["sector_3_time"] == pytest.approx(27.2)
        assert result["tyre_compound"] == "SOFT"
        assert result["tyre_age"] == 3.0
 
    def test_telemetry_statistics(self, session, telemetry_df):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
 
        assert result["mean_speed"] == pytest.approx(telemetry_df.Speed.mean())
        assert result["max_speed"] == pytest.approx(telemetry_df.Speed.max())
        assert result["speed_std"] == pytest.approx(telemetry_df.Speed.std())
        assert result["mean_rpm"] == pytest.approx(telemetry_df.RPM.mean())
        assert result["max_rpm"] == pytest.approx(telemetry_df.RPM.max())
        assert result["rpm_std"] == pytest.approx(telemetry_df.RPM.std())
        assert result["mean_throttle"] == pytest.approx(telemetry_df.Throttle.mean())
        assert result["mean_gear"] == pytest.approx(telemetry_df.nGear.mean())
        assert result["rpm_speed_corr"] == pytest.approx(telemetry_df.RPM.corr(telemetry_df.Speed))
        assert result["throttle_speed_corr"] == pytest.approx(
            telemetry_df.Throttle.corr(telemetry_df.Speed)
        )
 
    def test_geometry_features_come_from_get_diff_geom_feats(self, session, monkeypatch):
        monkeypatch.setattr(feat, "get_diff_geom_feats", lambda tele: (0.1, 0.2, 0.3))
 
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
 
        assert result["speed_curvature_corr"] == 0.1
        assert result["mean_curvature"] == 0.2
        assert result["max_curvature"] == 0.3
 
    def test_global_feats_are_merged_in(self, session):
        result = feat.get_driver_feats("16", session, GLOBAL_FEATS)
 
        for key, value in GLOBAL_FEATS.items():
            assert result[key] == value
 
    def test_looks_up_the_requested_driver(self, session):
        feat.get_driver_feats("44", session, GLOBAL_FEATS)
        session.laps.pick_drivers.assert_called_once_with("44")
 
    def test_returns_none_when_driver_has_no_laps(self, session, capsys):
        session.laps.pick_drivers.return_value.empty = True
 
        assert feat.get_driver_feats("16", session, GLOBAL_FEATS) is None
        assert "returning None" in capsys.readouterr().out
        session.laps.pick_drivers.return_value.pick_fastest.assert_not_called()
 
    def test_returns_none_when_telemetry_attribute_is_missing(self, session, fastest_lap, capsys):
        # A PropertyMock raising AttributeError doesn't work here: MagicMock
        # falls back to __getattr__ and returns a child mock. Deleting the
        # attribute makes access raise AttributeError for real.
        del fastest_lap.telemetry
 
        assert feat.get_driver_feats("16", session, GLOBAL_FEATS) is None
        assert "telemetry" in capsys.readouterr().out
 
    def test_returns_none_when_telemetry_raises_key_error(self, session, fastest_lap, capsys):
        type(fastest_lap).telemetry = PropertyMock(side_effect=KeyError("Speed"))
 
        assert feat.get_driver_feats("16", session, GLOBAL_FEATS) is None
        assert "telemetry" in capsys.readouterr().out
 
 
# ---------------------------------------------------------------------------
# get_all_drivers_feats
# ---------------------------------------------------------------------------
 
class TestGetAllDriversFeats:
    def test_calls_get_driver_feats_for_every_unique_driver(self, monkeypatch):
        sesh = MagicMock()
        sesh.name = "Practice 1"
        sesh.laps.DriverNumber.unique.return_value = np.array(["1", "44"])
        spy = MagicMock(side_effect=lambda d, s, g, season: {"driver": d})
        monkeypatch.setattr(feat, "get_driver_feats", spy)
 
        result = feat.get_all_drivers_feats(sesh, GLOBAL_FEATS, 2026)
 
        assert result == [{"driver": "1"}, {"driver": "44"}]
        assert spy.call_count == 2
        spy.assert_any_call("1", sesh, GLOBAL_FEATS, 2026)
        spy.assert_any_call("44", sesh, GLOBAL_FEATS, 2026)
 
    def test_keeps_none_entries_for_drivers_without_data(self, monkeypatch):
        # Filtering Nones is the caller's job (generate_features_per_session)
        sesh = MagicMock()
        sesh.laps.DriverNumber.unique.return_value = np.array(["1", "44"])
        monkeypatch.setattr(
            feat, "get_driver_feats",
            lambda d, s, g, season: None if d == "44" else {"driver": d},
        )
 
        result = feat.get_all_drivers_feats(sesh, GLOBAL_FEATS, 2026)
 
        assert result == [{"driver": "1"}, None]
 
    def test_no_drivers_returns_empty_list(self, monkeypatch):
        sesh = MagicMock()
        sesh.laps.DriverNumber.unique.return_value = np.array([])
        spy = MagicMock()
        monkeypatch.setattr(feat, "get_driver_feats", spy)
 
        assert feat.get_all_drivers_feats(sesh, GLOBAL_FEATS, 2026) == []
        spy.assert_not_called()
 
 
# ---------------------------------------------------------------------------
# get_global_features
# ---------------------------------------------------------------------------
 
class TestGetGlobalFeatures:
    @pytest.fixture
    def weather_session(self):
        sesh = MagicMock()
        sesh.laps.pick_fastest.return_value.LapTime = pd.Timedelta(seconds=88.0)
        sesh.weather_data = pd.DataFrame(
            {"AirTemp": [20.0, 25.0, 30.0], "TrackTemp": [30.0, 40.0, 50.0]}
        )
        return sesh
 
    def test_fastest_lap_in_seconds(self, weather_session):
        assert feat.get_global_features(weather_session)["fastest_lap"] == pytest.approx(88.0)
 
    def test_air_temperature_stats(self, weather_session):
        result = feat.get_global_features(weather_session)
 
        assert result["air_temp_mean"] == pytest.approx(25.0)
        assert result["air_temp_max"] == 30.0
        assert result["air_temp_min"] == 20.0
 
    def test_track_temperature_stats(self, weather_session):
        result = feat.get_global_features(weather_session)
 
        assert result["track_temp_mean"] == pytest.approx(40.0)
        assert result["track_temp_max"] == 50.0
        assert result["track_temp_min"] == 30.0
 
    def test_returns_exactly_the_expected_keys(self, weather_session):
        assert set(feat.get_global_features(weather_session)) == {
            "fastest_lap",
            "air_temp_mean", "air_temp_max", "air_temp_min",
            "track_temp_mean", "track_temp_max", "track_temp_min",
        }
 
 
# ---------------------------------------------------------------------------
# generate_features_per_session
# ---------------------------------------------------------------------------
 
class TestGenerateFeaturesPerSession:
    @pytest.fixture
    def patched(self, monkeypatch):
        """Patch FastF1's get_session plus the two helper functions."""
        fp_sesh = MagicMock()
        gs = MagicMock(return_value=fp_sesh)
        global_feats = MagicMock(return_value=GLOBAL_FEATS)
        all_feats = MagicMock(
            return_value=[
                {"driver": "1", "lap_time": 90.0},
                None,  # driver without data -> must be filtered out
                {"driver": "44", "lap_time": 91.0},
            ]
        )
        monkeypatch.setattr(feat, "gs", gs)
        monkeypatch.setattr(feat, "get_global_features", global_feats)
        monkeypatch.setattr(feat, "get_all_drivers_feats", all_feats)
        return {"sesh": fp_sesh, "gs": gs, "global": global_feats, "all": all_feats}
 
    def test_loads_the_requested_session(self, patched):
        feat.generate_features_per_session("Monaco Grand Prix", "FP2", 2025)
 
        patched["gs"].assert_called_once_with(2025, "Monaco Grand Prix", "FP2")
        patched["sesh"].load.assert_called_once()
 
    def test_season_defaults_to_module_constant(self, patched):
        feat.generate_features_per_session("Monaco Grand Prix", "FP1")
        patched["gs"].assert_called_once_with(feat.SEASON, "Monaco Grand Prix", "FP1")
 
    def test_passes_global_feats_down_to_driver_features(self, patched):
        feat.generate_features_per_session("Monaco Grand Prix", "FP1", 2025)
 
        patched["global"].assert_called_once_with(patched["sesh"])
        patched["all"].assert_called_once_with(patched["sesh"], GLOBAL_FEATS, 2025)
 
    def test_filters_none_and_adds_session_column(self, patched):
        df = feat.generate_features_per_session("Monaco Grand Prix", "FP3", 2025)
 
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["driver"]) == ["1", "44"]
        assert (df["session"] == "FP3").all()
 
    def test_all_drivers_missing_returns_empty_frame_with_session_column(self, patched):
        patched["all"].return_value = [None, None]
 
        df = feat.generate_features_per_session("Monaco Grand Prix", "FP1", 2025)
 
        assert df.empty
        assert "session" in df.columns
 
 
# ---------------------------------------------------------------------------
# generate_features_per_race
# ---------------------------------------------------------------------------
 
def _session_frame(session_name: str) -> pd.DataFrame:
    return pd.DataFrame({"driver": ["1", "44"], "session": [session_name] * 2})
 
 
class TestGenerateFeaturesPerRace:
    def test_concatenates_all_sessions_with_fresh_index(self, monkeypatch):
        spy = MagicMock(side_effect=lambda race, s, season: _session_frame(s))
        monkeypatch.setattr(feat, "generate_features_per_session", spy)
 
        df = feat.generate_features_per_race("Monaco Grand Prix", 2025)
 
        assert len(df) == 2 * len(feat.SESSIONS)
        assert list(df["session"].unique()) == feat.SESSIONS
        assert list(df.index) == list(range(len(df)))
 
    def test_calls_per_session_for_each_session_in_order(self, monkeypatch):
        spy = MagicMock(side_effect=lambda race, s, season: _session_frame(s))
        monkeypatch.setattr(feat, "generate_features_per_session", spy)
 
        feat.generate_features_per_race("Monaco Grand Prix", 2025)
 
        assert [c.args for c in spy.call_args_list] == [
            ("Monaco Grand Prix", "FP1", 2025),
            ("Monaco Grand Prix", "FP2", 2025),
            ("Monaco Grand Prix", "FP3", 2025),
        ]
 
    def test_skips_sessions_that_raise_value_error(self, monkeypatch, capsys):
        # e.g. sprint weekends have no FP2/FP3
        def fake(race, s, season):
            if s != "FP1":
                raise ValueError("session does not exist")
            return _session_frame(s)
 
        monkeypatch.setattr(feat, "generate_features_per_session", fake)
 
        df = feat.generate_features_per_race("Sprint GP", 2025)
 
        assert set(df["session"]) == {"FP1"}
        out = capsys.readouterr().out
        assert "Skipping FP2" in out and "Skipping FP3" in out
 
    def test_returns_empty_frame_when_every_session_fails(self, monkeypatch):
        def always_fail(race, s, season):
            raise ValueError("nope")
 
        monkeypatch.setattr(feat, "generate_features_per_session", always_fail)
 
        df = feat.generate_features_per_race("Cancelled GP", 2025)
 
        assert isinstance(df, pd.DataFrame)
        assert df.empty
 
    def test_other_exceptions_are_not_swallowed(self, monkeypatch):
        def boom(race, s, season):
            raise RuntimeError("network down")
 
        monkeypatch.setattr(feat, "generate_features_per_session", boom)
 
        with pytest.raises(RuntimeError, match="network down"):
            feat.generate_features_per_race("Monaco Grand Prix", 2025)
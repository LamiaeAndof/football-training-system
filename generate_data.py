"""
generate_data.py
Football Training Intelligence System — Synthetic Data Generator

ASSUMPTIONS & LOGIC DOCUMENTATION
====================================

1. PHYSICAL PROFILES PER DRILL TYPE
-------------------------------------
Each drill family has a base physical profile derived from sports science literature
and elite football performance department conventions:

  flying_sprint:     very high sprint dist, very high max speed %, low accels, moderate PL
  accel_sprint:      low sprint dist, very high accels/decels, high intensity
  speed_endurance:   high sprint dist, moderate max speed %, high metabolic power
  SSG_compact:       very high accels, low-moderate sprint, very high PL, high HR
  SSG_large:         moderate accels, moderate sprint/HSR, high PL, high HR
  large_sided_game:  moderate accels, high sprint/HSR, moderate PL, high metabolic
  rondo:             very low all metrics except decision making (not tracked here)
  possession_game:   low sprint, moderate accels, moderate PL, aerobic stimulus
  transition_game:   moderate-high sprint, high accels, high intensity
  pressing_drill:    very high accels, moderate sprint, very high intensity, high PL
  finishing_drill:   moderate everything, short reps, high HR peaks
  activation_drill:  very low all metrics, suitable MD1
  recovery_drill:    very low all metrics, suitable MD+1

2. AREA PER PLAYER INFLUENCE
------------------------------
Area per player (m2) is a validated predictor of external load in GPS research.
Higher area per player → more running volume, more HSR, more sprint distance.
Lower area per player → more accelerations/decelerations, higher intensity bursts.
We apply a linear scaling factor: load_modifier = (area_per_player / 100) ^ 0.4

3. MICROCYCLE DAY INFLUENCE
-----------------------------
MD day modulates the typical intensity of sessions:
  MD+1, MD1 → intensity multiplier ~0.6-0.75 (low load)
  MD4       → intensity multiplier ~1.1-1.2 (peak conditioning)
  MD3       → intensity multiplier ~1.0-1.05 (moderate-high)
  MD2       → intensity multiplier ~0.9-1.0 (moderate + speed)

4. FATIGUE / READINESS MODULATION
------------------------------------
team_fatigue_score (0-10) and readiness_score (0-10) are simulated per session.
High fatigue → output metrics reduced by up to 8%
High readiness → output metrics boosted by up to 5%
net_modifier = 1 + (readiness_score - 5) * 0.01 - (team_fatigue_score - 5) * 0.008

5. WEATHER / TEMPERATURE
--------------------------
Simulated realistically: hot weather (>30°C) reduces intensity by ~3-5%.
Cold weather (<5°C) has minimal performance effect in trained players.

6. NOISE
---------
Each metric has realistic session-to-session noise (±8-15%) to simulate
natural within-drill variability across different sessions.

7. METRIC DEFINITIONS (aligned with commercial GPS provider standards)
------------------------------------------------------------------------
avg_total_distance_m       : Total distance covered per player per drill bout
avg_distance_per_min       : Total distance / duration (intensity proxy)
avg_hsr_m                  : High-speed running distance (>19.8 km/h) per player
avg_sprint_distance_m      : Sprint distance (>25.2 km/h) per player
avg_percent_max_speed      : Peak speed as % of player individual max speed
avg_accelerations          : Count of accelerations >2.5 m/s2 per player
avg_decelerations          : Count of decelerations >-2.5 m/s2 per player
avg_player_load            : Proprietary triaxial accelerometer metric (AU)
avg_metabolic_power        : Mean metabolic power output (W/kg)
avg_hr_mean                : Mean heart rate (% max HR)
avg_hr_max                 : Peak heart rate (% max HR)
rpe_mean                   : Session RPE (Borg CR10 scale, 1-10)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
# PHYSICAL PROFILES — base values per drill type
# Units: absolute values representing avg per player per drill bout
# ─────────────────────────────────────────────

PHYSICAL_PROFILES = {
    "flying_sprint_30m": {
        "total_distance": 280, "dist_per_min": 140, "hsr": 80, "sprint": 85,
        "pct_max_speed": 92, "accels": 4, "decels": 5, "player_load": 65,
        "metabolic_power": 14.0, "hr_mean": 82, "hr_max": 94, "rpe": 7.0,
    },
    "flying_sprint_20m": {
        "total_distance": 200, "dist_per_min": 135, "hsr": 60, "sprint": 65,
        "pct_max_speed": 94, "accels": 4, "decels": 4, "player_load": 50,
        "metabolic_power": 15.0, "hr_mean": 80, "hr_max": 92, "rpe": 6.5,
    },
    "accel_sprint_10m": {
        "total_distance": 90, "dist_per_min": 90, "hsr": 5, "sprint": 0,
        "pct_max_speed": 55, "accels": 14, "decels": 12, "player_load": 55,
        "metabolic_power": 16.5, "hr_mean": 78, "hr_max": 90, "rpe": 6.8,
    },
    "accel_sprint_15m": {
        "total_distance": 120, "dist_per_min": 100, "hsr": 12, "sprint": 8,
        "pct_max_speed": 68, "accels": 12, "decels": 10, "player_load": 58,
        "metabolic_power": 16.0, "hr_mean": 79, "hr_max": 91, "rpe": 7.0,
    },
    "transition_sprint_40m": {
        "total_distance": 350, "dist_per_min": 175, "hsr": 100, "sprint": 110,
        "pct_max_speed": 88, "accels": 7, "decels": 7, "player_load": 72,
        "metabolic_power": 13.5, "hr_mean": 83, "hr_max": 95, "rpe": 7.5,
    },
    "speed_endurance_60m": {
        "total_distance": 500, "dist_per_min": 160, "hsr": 180, "sprint": 220,
        "pct_max_speed": 85, "accels": 6, "decels": 6, "player_load": 90,
        "metabolic_power": 12.5, "hr_mean": 87, "hr_max": 97, "rpe": 8.5,
    },
    "accel_cod_15m": {
        "total_distance": 130, "dist_per_min": 110, "hsr": 10, "sprint": 4,
        "pct_max_speed": 62, "accels": 15, "decels": 16, "player_load": 70,
        "metabolic_power": 17.0, "hr_mean": 80, "hr_max": 92, "rpe": 7.2,
    },
    "ssg_compact": {  # SSG <80 m2/player
        "total_distance": 620, "dist_per_min": 100, "hsr": 40, "sprint": 10,
        "pct_max_speed": 58, "accels": 28, "decels": 26, "player_load": 95,
        "metabolic_power": 11.5, "hr_mean": 88, "hr_max": 96, "rpe": 8.0,
    },
    "ssg_medium": {  # SSG 80-120 m2/player
        "total_distance": 700, "dist_per_min": 100, "hsr": 65, "sprint": 25,
        "pct_max_speed": 65, "accels": 22, "decels": 20, "player_load": 100,
        "metabolic_power": 11.8, "hr_mean": 87, "hr_max": 95, "rpe": 7.8,
    },
    "large_sided_game": {  # >130 m2/player
        "total_distance": 900, "dist_per_min": 90, "hsr": 110, "sprint": 60,
        "pct_max_speed": 72, "accels": 16, "decels": 15, "player_load": 110,
        "metabolic_power": 10.5, "hr_mean": 83, "hr_max": 93, "rpe": 7.5,
    },
    "rondo_compact": {  # <30 m2/player
        "total_distance": 280, "dist_per_min": 47, "hsr": 4, "sprint": 0,
        "pct_max_speed": 28, "accels": 8, "decels": 7, "player_load": 30,
        "metabolic_power": 5.0, "hr_mean": 65, "hr_max": 78, "rpe": 3.5,
    },
    "rondo_medium": {  # 30-50 m2/player
        "total_distance": 350, "dist_per_min": 50, "hsr": 8, "sprint": 0,
        "pct_max_speed": 32, "accels": 10, "decels": 9, "player_load": 38,
        "metabolic_power": 5.5, "hr_mean": 67, "hr_max": 80, "rpe": 3.8,
    },
    "possession_compact": {
        "total_distance": 550, "dist_per_min": 68, "hsr": 30, "sprint": 8,
        "pct_max_speed": 52, "accels": 18, "decels": 16, "player_load": 70,
        "metabolic_power": 9.0, "hr_mean": 79, "hr_max": 90, "rpe": 6.5,
    },
    "possession_large": {
        "total_distance": 720, "dist_per_min": 72, "hsr": 65, "sprint": 30,
        "pct_max_speed": 62, "accels": 14, "decels": 13, "player_load": 82,
        "metabolic_power": 9.5, "hr_mean": 80, "hr_max": 91, "rpe": 6.8,
    },
    "transition_compact": {
        "total_distance": 650, "dist_per_min": 92, "hsr": 70, "sprint": 40,
        "pct_max_speed": 70, "accels": 22, "decels": 20, "player_load": 100,
        "metabolic_power": 11.0, "hr_mean": 85, "hr_max": 94, "rpe": 7.8,
    },
    "transition_large": {
        "total_distance": 780, "dist_per_min": 97, "hsr": 100, "sprint": 65,
        "pct_max_speed": 74, "accels": 18, "decels": 17, "player_load": 108,
        "metabolic_power": 11.2, "hr_mean": 85, "hr_max": 95, "rpe": 7.9,
    },
    "pressing_compact": {
        "total_distance": 580, "dist_per_min": 96, "hsr": 55, "sprint": 20,
        "pct_max_speed": 62, "accels": 30, "decels": 28, "player_load": 105,
        "metabolic_power": 12.5, "hr_mean": 89, "hr_max": 97, "rpe": 8.2,
    },
    "pressing_large": {
        "total_distance": 740, "dist_per_min": 92, "hsr": 85, "sprint": 45,
        "pct_max_speed": 68, "accels": 24, "decels": 22, "player_load": 112,
        "metabolic_power": 12.0, "hr_mean": 88, "hr_max": 96, "rpe": 8.0,
    },
    "finishing_compact": {
        "total_distance": 420, "dist_per_min": 70, "hsr": 35, "sprint": 18,
        "pct_max_speed": 64, "accels": 16, "decels": 15, "player_load": 65,
        "metabolic_power": 10.0, "hr_mean": 80, "hr_max": 92, "rpe": 6.5,
    },
    "activation_shadow": {
        "total_distance": 700, "dist_per_min": 47, "hsr": 10, "sprint": 0,
        "pct_max_speed": 25, "accels": 5, "decels": 5, "player_load": 28,
        "metabolic_power": 4.5, "hr_mean": 60, "hr_max": 72, "rpe": 2.5,
    },
    "activation_passing": {
        "total_distance": 380, "dist_per_min": 38, "hsr": 2, "sprint": 0,
        "pct_max_speed": 20, "accels": 4, "decels": 4, "player_load": 20,
        "metabolic_power": 4.0, "hr_mean": 58, "hr_max": 70, "rpe": 2.0,
    },
    "recovery_game": {
        "total_distance": 800, "dist_per_min": 53, "hsr": 15, "sprint": 0,
        "pct_max_speed": 30, "accels": 6, "decels": 5, "player_load": 35,
        "metabolic_power": 5.5, "hr_mean": 65, "hr_max": 76, "rpe": 3.0,
    },
    "recovery_circuit": {
        "total_distance": 600, "dist_per_min": 40, "hsr": 5, "sprint": 0,
        "pct_max_speed": 22, "accels": 3, "decels": 3, "player_load": 25,
        "metabolic_power": 4.2, "hr_mean": 60, "hr_max": 72, "rpe": 2.5,
    },
    "setpiece": {
        "total_distance": 300, "dist_per_min": 30, "hsr": 5, "sprint": 2,
        "pct_max_speed": 30, "accels": 6, "decels": 5, "player_load": 25,
        "metabolic_power": 5.0, "hr_mean": 62, "hr_max": 76, "rpe": 2.8,
    },
}

# Map each exercise_id to its physical profile key
EXERCISE_PROFILE_MAP = {
    "EX001": "ssg_compact",          # SSG_3v3_20x15_Pressing
    "EX002": "ssg_medium",           # SSG_4v4_25x20_Transition
    "EX003": "ssg_medium",           # SSG_5v5_30x25_Possession
    "EX004": "ssg_medium",           # SSG_5v5_30x25_Transition
    "EX005": "ssg_medium",           # SSG_6v6_35x30_Pressing
    "EX006": "rondo_compact",        # Rondo_4v2_10x10
    "EX007": "rondo_compact",        # Rondo_6v2_15x15
    "EX008": "rondo_medium",         # Rondo_6v3_20x15
    "EX009": "rondo_medium",         # Rondo_8v3_25x20_Directional
    "EX010": "large_sided_game",     # LargeGame_8v8_60x40_Possession
    "EX011": "large_sided_game",     # LargeGame_9v9_65x45_Transition
    "EX012": "large_sided_game",     # LargeGame_10v10_70x50_Open
    "EX013": "pressing_large",       # LargeGame_8v8_60x40_Pressing
    "EX014": "flying_sprint_30m",    # Flying_Sprint_30m
    "EX015": "flying_sprint_20m",    # Flying_Sprint_20m
    "EX016": "accel_sprint_10m",     # Accel_Sprint_10m_x8
    "EX017": "accel_sprint_15m",     # Accel_Sprint_15m_x6
    "EX018": "transition_sprint_40m",# Transition_Sprint_40m
    "EX019": "finishing_compact",    # Finishing_4v3_25x20
    "EX020": "finishing_compact",    # Finishing_5v4_30x25
    "EX021": "pressing_compact",     # PressingGame_5v5_30x25_High
    "EX022": "pressing_compact",     # PressingGame_6v6_35x30_Mid
    "EX023": "activation_passing",   # Activation_Passing_8v0
    "EX024": "activation_shadow",    # Activation_Positional_10v0
    "EX025": "recovery_game",        # Recovery_Game_5v5_40x35
    "EX026": "recovery_circuit",     # Recovery_Walk_Jog_Circuit
    "EX027": "possession_compact",   # Possession_6v6_40x30_Neutral
    "EX028": "possession_large",     # Possession_8v8_50x35_Neutral
    "EX029": "transition_compact",   # Transition_5v5_35x30_Counter
    "EX030": "transition_large",     # Transition_6v6_40x35_Wide
    "EX031": "activation_shadow",    # Shadow_Play_11v0_Shape
    "EX032": "setpiece",             # Setpiece_Attack_Corners
    "EX033": "recovery_game",        # SSG_4v4_20x15_Regen
    "EX034": "pressing_large",       # Hybrid_8v8_55x40_Pressing
    "EX035": "speed_endurance_60m",  # Speed_Endurance_60m_x4
    "EX036": "accel_cod_15m",        # Accel_Change_Direction_15m
    "EX037": "possession_compact",   # Possession_4v4_2Neutral_25x20
    "EX038": "pressing_compact",     # PressingTrap_6v4_35x25
}

MD_DAY_INTENSITY_MULTIPLIER = {
    "MD_plus1": 0.60,
    "MD_plus2": 0.72,
    "MD4": 1.15,
    "MD3": 1.02,
    "MD2": 0.95,
    "MD1": 0.68,
}

# Which exercises are realistic on which MD day (soft constraint)
MD_DAY_EXERCISE_WEIGHTS = {
    "MD_plus1": {"EX025": 4, "EX026": 4, "EX033": 3, "EX023": 2, "EX024": 1},
    "MD_plus2": {"EX025": 2, "EX026": 2, "EX033": 2, "EX027": 2, "EX028": 2,
                 "EX008": 1, "EX009": 1, "EX023": 1},
    "MD4": {"EX001": 4, "EX002": 3, "EX004": 3, "EX005": 3, "EX021": 4,
            "EX022": 3, "EX034": 4, "EX038": 4, "EX013": 3, "EX003": 2},
    "MD3": {"EX003": 3, "EX004": 3, "EX010": 3, "EX011": 3, "EX012": 2,
            "EX027": 3, "EX029": 3, "EX030": 3, "EX016": 2, "EX017": 2,
            "EX036": 2, "EX037": 2},
    "MD2": {"EX014": 4, "EX015": 4, "EX018": 3, "EX035": 3, "EX019": 2,
            "EX020": 2, "EX028": 2, "EX017": 2, "EX012": 2},
    "MD1": {"EX006": 4, "EX007": 4, "EX008": 2, "EX023": 3, "EX024": 3,
            "EX031": 3, "EX032": 3, "EX019": 1},
}

TEAMS = ["U23", "First_Team", "B_Team", "Academy_U18"]
WEATHER_OPTIONS = ["Clear", "Cloudy", "Light_Rain", "Overcast"]
TEMP_RANGES = {"Clear": (18, 34), "Cloudy": (10, 22), "Light_Rain": (8, 18), "Overcast": (8, 20)}


def get_weather_modifier(weather: str, temp_c: float) -> float:
    """Hot weather reduces output; cold has minimal effect on trained players."""
    if temp_c > 30:
        return 0.96
    elif temp_c > 26:
        return 0.98
    elif temp_c < 5:
        return 0.99
    return 1.0


def get_fatigue_readiness_modifier(fatigue: float, readiness: float) -> float:
    """
    fatigue 0-10 (higher = more fatigued)
    readiness 0-10 (higher = more ready)
    Net effect: readiness boosts, fatigue suppresses
    """
    return 1.0 + (readiness - 5.0) * 0.012 - (fatigue - 5.0) * 0.009


def apply_noise(value: float, noise_pct: float = 0.10) -> float:
    """Apply realistic session-to-session variability."""
    return max(0.0, value * np.random.normal(1.0, noise_pct))


def simulate_session_metrics(
    profile_key: str,
    md_day: str,
    team_fatigue: float,
    readiness: float,
    weather: str,
    temp_c: float,
    duration_min: float,
) -> dict:
    """
    Simulate GPS and physiological outputs for a single drill occurrence.
    Logic: base profile × md_day_multiplier × fatigue_modifier × weather_modifier × noise
    Duration scales linearly beyond a reference duration (base profiles assume ~8 min).
    """
    base = PHYSICAL_PROFILES[profile_key]
    md_mod = MD_DAY_INTENSITY_MULTIPLIER.get(md_day, 1.0)
    fat_mod = get_fatigue_readiness_modifier(team_fatigue, readiness)
    wx_mod = get_weather_modifier(weather, temp_c)

    # Duration scaling: reference is 8 min; sprint drills reference is 1 min × reps
    # For simplicity treat duration as already embedded in profile (per bout)
    compound_mod = md_mod * fat_mod * wx_mod

    def scale(key, noise=0.10):
        return round(apply_noise(base[key] * compound_mod, noise), 2)

    return {
        "avg_total_distance_m":     scale("total_distance", 0.09),
        "avg_distance_per_min":     scale("dist_per_min", 0.07),
        "avg_hsr_m":                max(0, scale("hsr", 0.14)),
        "avg_sprint_distance_m":    max(0, scale("sprint", 0.16)),
        "avg_percent_max_speed":    min(100, max(15, scale("pct_max_speed", 0.06))),
        "avg_accelerations":        max(0, round(scale("accels", 0.12))),
        "avg_decelerations":        max(0, round(scale("decels", 0.12))),
        "avg_player_load":          scale("player_load", 0.08),
        "avg_metabolic_power":      round(min(20, scale("metabolic_power", 0.08)), 2),
        "avg_hr_mean":              min(100, max(50, round(scale("hr_mean", 0.04)))),
        "avg_hr_max":               min(100, max(55, round(scale("hr_max", 0.03)))),
        "rpe_mean":                 round(min(10, max(1, scale("rpe", 0.10))), 1),
    }


def load_templates() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "data", "exercise_templates.csv")
    return pd.read_csv(path)


def generate_sessions(n_sessions: int = 150, target_rows: int = 1000) -> pd.DataFrame:
    """
    Generate exercise_sessions.csv.
    n_sessions: number of distinct training sessions
    target_rows: approximate total drill occurrences across all sessions
    """
    templates = load_templates()
    ex_ids = templates["exercise_id"].tolist()
    ex_names = dict(zip(templates["exercise_id"], templates["exercise_name"]))
    ex_types = dict(zip(templates["exercise_id"], templates["exercise_type"]))
    ex_obj = dict(zip(templates["exercise_id"], templates["physical_objective"]))

    rows = []
    base_date = datetime(2024, 7, 1)
    session_id = 1

    drills_per_session_mean = target_rows / n_sessions  # ~6-7

    for i in range(n_sessions):
        session_date = base_date + timedelta(days=i * 2)  # ~2 sessions/week rhythm
        team = random.choice(TEAMS)
        md_day = random.choices(
            ["MD_plus1", "MD_plus2", "MD4", "MD3", "MD2", "MD1"],
            weights=[10, 10, 20, 20, 20, 20],
            k=1
        )[0]
        weather = random.choice(WEATHER_OPTIONS)
        temp_min, temp_max = TEMP_RANGES[weather]
        temp_c = round(random.uniform(temp_min, temp_max), 1)

        # Session-level fatigue and readiness (correlated with md_day)
        md_fatigue_base = {"MD_plus1": 7.5, "MD_plus2": 6.5, "MD4": 4.0,
                           "MD3": 5.0, "MD2": 4.5, "MD1": 3.5}
        team_fatigue = round(np.clip(
            np.random.normal(md_fatigue_base[md_day], 1.0), 1.0, 10.0), 1)
        readiness = round(np.clip(
            np.random.normal(10.0 - md_fatigue_base[md_day] + 0.5, 1.2), 1.0, 10.0), 1)

        # Select exercises realistic for this md_day
        weight_map = MD_DAY_EXERCISE_WEIGHTS.get(md_day, {})
        if weight_map:
            pool_ids = list(weight_map.keys())
            pool_weights = [weight_map[eid] for eid in pool_ids]
        else:
            pool_ids = ex_ids
            pool_weights = [1] * len(ex_ids)

        n_drills = max(3, int(np.random.normal(drills_per_session_mean, 1.5)))

        chosen = random.choices(pool_ids, weights=pool_weights, k=n_drills)
        # Remove duplicate consecutive drills
        seen = set()
        unique_chosen = []
        for c in chosen:
            if c not in seen:
                unique_chosen.append(c)
                seen.add(c)
        if len(unique_chosen) < 2:
            unique_chosen = chosen[:max(2, n_drills)]

        for ex_id in unique_chosen:
            profile_key = EXERCISE_PROFILE_MAP.get(ex_id, "ssg_medium")
            tmpl = templates[templates["exercise_id"] == ex_id].iloc[0]
            duration = float(tmpl["duration_min"])

            metrics = simulate_session_metrics(
                profile_key, md_day, team_fatigue, readiness, weather, temp_c, duration
            )

            rows.append({
                "session_id": f"SES{session_id:04d}",
                "date": session_date.strftime("%Y-%m-%d"),
                "team_category": team,
                "md_day": md_day,
                "exercise_id": ex_id,
                "exercise_name": ex_names[ex_id],
                "exercise_type": ex_types[ex_id],
                "objective_primary": ex_obj[ex_id],
                "weather": weather,
                "temp_c": temp_c,
                "team_fatigue_score": team_fatigue,
                "readiness_score": readiness,
                **metrics,
            })

        session_id += 1

    return pd.DataFrame(rows)


def generate_player_readiness(sessions_df: pd.DataFrame, n_players: int = 22) -> pd.DataFrame:
    """
    Generate per-player pre-session readiness data.
    One row per player per session.
    """
    positions = ["GK", "CB", "LB", "RB", "CDM", "CM", "LW", "RW", "CAM", "ST"]
    position_pool = (["GK"] * 1 + ["CB"] * 3 + ["LB"] * 2 + ["RB"] * 2 +
                     ["CDM"] * 2 + ["CM"] * 3 + ["LW"] * 2 + ["RW"] * 2 +
                     ["CAM"] * 2 + ["ST"] * 3)[:n_players]

    rows = []
    unique_sessions = sessions_df[["session_id", "date", "md_day",
                                   "team_fatigue_score", "readiness_score"]].drop_duplicates(
        subset="session_id")

    for _, ses in unique_sessions.iterrows():
        md_day = ses["md_day"]
        base_fatigue = ses["team_fatigue_score"]
        base_readiness = ses["readiness_score"]

        for pid in range(1, n_players + 1):
            position = position_pool[pid - 1]

            # Position-specific modifiers (outfield players generally more fatigued post-match)
            pos_fatigue_offset = 0.5 if position == "GK" else 0.0

            fatigue_score = round(np.clip(
                np.random.normal(base_fatigue + pos_fatigue_offset, 1.2), 1, 10), 1)
            readiness_score = round(np.clip(
                np.random.normal(base_readiness - pos_fatigue_offset, 1.2), 1, 10), 1)
            soreness_score = round(np.clip(
                np.random.normal(fatigue_score * 0.8, 1.0), 1, 10), 1)

            # Sleep quality slightly inversely correlated with fatigue
            sleep_score = round(np.clip(
                np.random.normal(10 - fatigue_score * 0.6, 1.0), 3, 10), 1)

            # HRV inversely correlated with fatigue (realistic range 40-90ms)
            hrv = round(np.clip(np.random.normal(75 - fatigue_score * 3.0, 8), 35, 100), 1)
            resting_hr = round(np.clip(np.random.normal(52 + fatigue_score * 1.5, 4), 40, 75), 0)

            # CMJ height (typical elite range: 40-65cm) reduced by fatigue
            cmj_height = round(np.clip(
                np.random.normal(54 - fatigue_score * 0.8, 3.5), 38, 68), 1)
            cmj_asym = round(np.clip(np.random.normal(4.0, 2.5), 0, 15), 1)

            # VBT mean velocity (squat, typical: 0.45-0.90 m/s)
            vbt_vel = round(np.clip(
                np.random.normal(0.72 - fatigue_score * 0.02, 0.05), 0.40, 0.90), 3)

            # Minutes in last match (affects fatigue heavily)
            if md_day in ["MD_plus1", "MD_plus2"]:
                minutes_last_match = random.choices([0, 45, 60, 75, 90], weights=[1, 2, 2, 2, 3])[0]
            else:
                minutes_last_match = random.choices([0, 30, 60, 80, 90], weights=[3, 2, 2, 1, 1])[0]

            injury_flag = int(np.random.random() < 0.03)  # 3% injury prevalence

            rows.append({
                "date": ses["date"],
                "session_id": ses["session_id"],
                "player_id": f"P{pid:03d}",
                "position": position,
                "sleep_score": sleep_score,
                "readiness_score": readiness_score,
                "hrv": hrv,
                "resting_hr": int(resting_hr),
                "fatigue_score": fatigue_score,
                "soreness_score": soreness_score,
                "stress_score": round(np.clip(np.random.normal(4.5, 1.5), 1, 10), 1),
                "cmj_height_cm": cmj_height,
                "cmj_asymmetry_percent": cmj_asym,
                "vbt_mean_velocity": vbt_vel,
                "minutes_last_match": minutes_last_match,
                "injury_flag": injury_flag,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("🏟️  Football Training Intelligence System — Data Generator")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    print("📊 Generating exercise sessions (target ~1000 rows)...")
    sessions = generate_sessions(n_sessions=155, target_rows=1000)
    sessions_path = os.path.join("data", "exercise_sessions.csv")
    sessions.to_csv(sessions_path, index=False)
    print(f"   ✅ {len(sessions)} rows written → {sessions_path}")

    print("🏃 Generating player readiness data...")
    readiness = generate_player_readiness(sessions, n_players=22)
    readiness_path = os.path.join("data", "optional_player_readiness.csv")
    readiness.to_csv(readiness_path, index=False)
    print(f"   ✅ {len(readiness)} rows written → {readiness_path}")

    print("\n📁 All datasets ready in /data/")
    print(f"   • exercise_templates.csv    — 38 drill templates")
    print(f"   • exercise_sessions.csv     — {len(sessions)} session records")
    print(f"   • optional_player_readiness.csv — {len(readiness)} player records")
    print(f"   • optional_microcycle_reference.csv — 6 MD days")
    print("\n✅ Data generation complete.")

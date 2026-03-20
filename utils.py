"""
utils.py
Football Training Intelligence System — Utility Functions & Recommendation Engine

RECOMMENDATION ENGINE LOGIC
==============================

The engine works in 3 stages:

STAGE 1 — HARD FILTER
  Eliminate exercises that are fundamentally incompatible with the query:
  - Wrong MD day (if md_day is specified and exercise is not designed for it)
  - Wrong exercise type (e.g. recovery requested → only recovery/activation types pass)

STAGE 2 — SCORING
  Each candidate drill receives a score from 0-100 based on:
    - Intensity match         (weight: 0.25)
    - Sprint exposure match   (weight: 0.20)
    - Acceleration match      (weight: 0.20)
    - Objective match         (weight: 0.20)
    - MD day alignment        (weight: 0.15)
  Sub-scores are computed by comparing the drill's average observed profile
  (from sessions data) against query targets, using a Gaussian proximity function.

STAGE 3 — RANKING + EXPLANATION
  Top 5 drills are returned with:
    - Composite score
    - Per-dimension sub-scores
    - Natural language rationale string
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# PROFILE AGGREGATION
# ─────────────────────────────────────────────────────────────────

def build_drill_profiles(sessions_df: pd.DataFrame, templates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate session data to get mean physical profile per exercise.
    Merge with template metadata for full drill descriptor.
    """
    numeric_cols = [
        "avg_total_distance_m", "avg_distance_per_min", "avg_hsr_m",
        "avg_sprint_distance_m", "avg_percent_max_speed", "avg_accelerations",
        "avg_decelerations", "avg_player_load", "avg_metabolic_power",
        "avg_hr_mean", "avg_hr_max", "rpe_mean"
    ]
    profile = (sessions_df
               .groupby("exercise_id")[numeric_cols]
               .mean()
               .reset_index())

    merged = pd.merge(templates_df, profile, on="exercise_id", how="left")
    return merged


# ─────────────────────────────────────────────────────────────────
# QUERY PARSER
# ─────────────────────────────────────────────────────────────────

INTENSITY_LEVELS = {
    "very_low": 0.15, "low": 0.30, "moderate": 0.50,
    "moderate_high": 0.65, "high": 0.80, "very_high": 0.95,
}
EXPOSURE_LEVELS = {
    "none": 0.0, "very_low": 0.1, "low": 0.2, "moderate": 0.45,
    "high": 0.70, "very_high": 0.90,
}


def parse_query(
    objective: Optional[str] = None,
    intensity: Optional[str] = None,
    md_day: Optional[str] = None,
    sprint_need: Optional[str] = None,
    acceleration_need: Optional[str] = None,
    recovery_mode: bool = False,
    exercise_type_filter: Optional[str] = None,
) -> dict:
    """
    Normalize coach query inputs into a structured query dict.
    """
    return {
        "objective": objective,
        "intensity": INTENSITY_LEVELS.get(intensity) if intensity else None,
        "md_day": md_day,
        "sprint_need": EXPOSURE_LEVELS.get(sprint_need) if sprint_need else None,
        "acceleration_need": EXPOSURE_LEVELS.get(acceleration_need) if acceleration_need else None,
        "recovery_mode": recovery_mode,
        "exercise_type_filter": exercise_type_filter,
    }


# ─────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def gaussian_proximity(observed: float, target: float, sigma: float = 0.20) -> float:
    """
    Score 0-1 based on how close observed is to target (normalized 0-1 scale).
    Uses a Gaussian with width sigma. Returns 1.0 when observed == target.
    """
    if target is None:
        return 0.5  # neutral score when no target specified
    return float(np.exp(-0.5 * ((observed - target) / sigma) ** 2))


def normalize_metric(value: float, min_val: float, max_val: float) -> float:
    """Map a metric to [0,1] range for comparison."""
    if max_val == min_val:
        return 0.5
    return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))


# Column-level normalization bounds (realistic football GPS ranges)
METRIC_BOUNDS = {
    "avg_hr_mean":             (55, 95),
    "avg_hr_max":              (65, 100),
    "rpe_mean":                (1, 10),
    "avg_sprint_distance_m":   (0, 250),
    "avg_accelerations":       (0, 35),
    "avg_distance_per_min":    (30, 200),
    "avg_metabolic_power":     (4, 18),
    "avg_player_load":         (20, 120),
    "avg_hsr_m":               (0, 220),
    "avg_percent_max_speed":   (20, 100),
}


def score_drill(drill_row: pd.Series, query: dict, all_profiles: pd.DataFrame) -> dict:
    """
    Score a single drill against a coach query.
    Returns dict with sub-scores and composite score.
    """
    # ── Normalization helpers ──
    def norm(col):
        lo, hi = METRIC_BOUNDS.get(col, (0, 1))
        val = drill_row.get(col, np.nan)
        if pd.isna(val):
            return 0.3
        return normalize_metric(float(val), lo, hi)

    # ── 1. Intensity score ──
    intensity_proxy = (norm("avg_hr_mean") * 0.3 +
                       norm("rpe_mean") * 0.3 +
                       norm("avg_metabolic_power") * 0.2 +
                       norm("avg_distance_per_min") * 0.2)
    if query["intensity"] is not None:
        s_intensity = gaussian_proximity(intensity_proxy, query["intensity"], sigma=0.22)
    else:
        s_intensity = 0.5

    # ── 2. Sprint exposure score ──
    sprint_proxy = (norm("avg_sprint_distance_m") * 0.5 +
                    norm("avg_percent_max_speed") * 0.3 +
                    norm("avg_hsr_m") * 0.2)
    if query["sprint_need"] is not None:
        s_sprint = gaussian_proximity(sprint_proxy, query["sprint_need"], sigma=0.22)
    else:
        s_sprint = 0.5

    # ── 3. Acceleration score ──
    accel_proxy = norm("avg_accelerations") * 0.6 + norm("avg_decelerations") * 0.4
    if query["acceleration_need"] is not None:
        s_accel = gaussian_proximity(accel_proxy, query["acceleration_need"], sigma=0.22)
    else:
        s_accel = 0.5

    # ── 4. Objective / type match ──
    s_objective = 0.5
    obj_q = (query.get("objective") or "").lower()
    phys_obj = str(drill_row.get("physical_objective", "")).lower()
    tac_obj = str(drill_row.get("tactical_objective", "")).lower()
    ex_type = str(drill_row.get("exercise_type", "")).lower()

    obj_keywords = {
        "aerobic": ["aerobic", "possession", "large_sided"],
        "speed": ["speed_development", "sprint", "sprint_drill"],
        "acceleration": ["acceleration_development", "sprint_drill", "pressing"],
        "recovery": ["recovery", "activation"],
        "activation": ["activation", "recovery"],
        "intensity": ["high_intensity", "pressing", "transition", "ssg"],
        "pressing": ["pressing", "high_intensity"],
        "possession": ["possession", "aerobic", "rondo"],
        "transition": ["transition", "mixed"],
    }

    for kw, targets in obj_keywords.items():
        if kw in obj_q:
            if any(t in phys_obj or t in ex_type or t in tac_obj for t in targets):
                s_objective = 0.90
                break
            else:
                s_objective = 0.15

    # Type filter (hard preference)
    if query["exercise_type_filter"] and query["exercise_type_filter"] != "all":
        if query["exercise_type_filter"].lower() in ex_type:
            s_objective = min(1.0, s_objective + 0.15)
        else:
            s_objective = max(0.0, s_objective - 0.30)

    # ── 5. MD day alignment ──
    recommended_md = str(drill_row.get("recommended_md_day", "")).strip()
    query_md = (query.get("md_day") or "").strip()
    if query_md:
        if query_md == recommended_md:
            s_md = 1.0
        elif _md_compatible(query_md, recommended_md):
            s_md = 0.65
        else:
            s_md = 0.10
    else:
        s_md = 0.5

    # Recovery mode override
    if query["recovery_mode"]:
        if ex_type in ("recovery_drill", "activation_drill"):
            s_intensity = 1.0
            s_sprint = 1.0
            s_accel = 1.0
        else:
            s_intensity *= 0.3

    # ── Composite ──
    weights = {"intensity": 0.25, "sprint": 0.20, "accel": 0.20,
               "objective": 0.20, "md": 0.15}
    composite = (s_intensity * weights["intensity"] +
                 s_sprint * weights["sprint"] +
                 s_accel * weights["accel"] +
                 s_objective * weights["objective"] +
                 s_md * weights["md"])

    return {
        "score_intensity": round(s_intensity * 100, 1),
        "score_sprint": round(s_sprint * 100, 1),
        "score_accel": round(s_accel * 100, 1),
        "score_objective": round(s_objective * 100, 1),
        "score_md": round(s_md * 100, 1),
        "composite_score": round(composite * 100, 1),
    }


def _md_compatible(query_md: str, recommended_md: str) -> bool:
    """
    Adjacent MD days are partially compatible.
    MD3 and MD4 are neighbors; MD2 and MD3 are neighbors; etc.
    """
    order = ["MD_plus1", "MD_plus2", "MD4", "MD3", "MD2", "MD1"]
    if query_md in order and recommended_md in order:
        return abs(order.index(query_md) - order.index(recommended_md)) == 1
    return False


# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────

def recommend_drills(
    profiles_df: pd.DataFrame,
    objective: Optional[str] = None,
    intensity: Optional[str] = None,
    md_day: Optional[str] = None,
    sprint_need: Optional[str] = None,
    acceleration_need: Optional[str] = None,
    recovery_mode: bool = False,
    exercise_type_filter: Optional[str] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Main recommendation function.
    Returns top_n drills with scores and rationale.
    """
    query = parse_query(
        objective=objective,
        intensity=intensity,
        md_day=md_day,
        sprint_need=sprint_need,
        acceleration_need=acceleration_need,
        recovery_mode=recovery_mode,
        exercise_type_filter=exercise_type_filter,
    )

    results = []
    for _, row in profiles_df.iterrows():
        scores = score_drill(row, query, profiles_df)
        results.append({**row.to_dict(), **scores})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("composite_score", ascending=False)
    top = results_df.head(top_n).copy()
    top["rationale"] = top.apply(lambda r: build_rationale(r, query), axis=1)
    return top.reset_index(drop=True)


def build_rationale(row: pd.Series, query: dict) -> str:
    """
    Generate a human-readable rationale string for a recommended drill.
    """
    parts = []

    # Drill type context
    ex_type = str(row.get("exercise_type", "")).replace("_", " ")
    parts.append(f"**{ex_type.title()}** with {int(row.get('players_total', 0) or 0)} players "
                 f"on {row.get('pitch_length_m', '?')}×{row.get('pitch_width_m', '?')} m.")

    # Physical highlights
    sprint_m = row.get("avg_sprint_distance_m", 0) or 0
    hsr_m = row.get("avg_hsr_m", 0) or 0
    accels = row.get("avg_accelerations", 0) or 0
    hr_mean = row.get("avg_hr_mean", 0) or 0
    rpe = row.get("rpe_mean", 0) or 0
    pl = row.get("avg_player_load", 0) or 0

    if sprint_m > 50:
        parts.append(f"High sprint exposure (~{sprint_m:.0f}m per player).")
    elif sprint_m > 15:
        parts.append(f"Moderate sprint distance (~{sprint_m:.0f}m per player).")
    else:
        parts.append(f"Low sprint distance (~{sprint_m:.0f}m) — not a speed drill.")

    if accels > 20:
        parts.append(f"Very high acceleration count (~{accels:.0f}/player) — COD-intensive.")
    elif accels > 12:
        parts.append(f"High acceleration demand (~{accels:.0f}/player).")

    if hr_mean > 85:
        parts.append(f"High cardiovascular stimulus (HR ~{hr_mean:.0f}% max).")
    elif hr_mean < 70:
        parts.append(f"Low cardiac load (HR ~{hr_mean:.0f}% max) — suitable for recovery.")

    parts.append(f"Mean RPE {rpe:.1f}/10, Player Load {pl:.0f} AU.")

    # MD day fit
    rec_md = str(row.get("recommended_md_day", "")).replace("_", "-")
    query_md = (query.get("md_day") or "").replace("_", "-")
    if query_md and query_md in rec_md:
        parts.append(f"✅ Designed for {rec_md} — perfect day alignment.")
    elif query_md:
        parts.append(f"ℹ️ Typically used on {rec_md}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────

def get_intensity_label(rpe: float, hr: float) -> str:
    combined = (rpe / 10) * 0.5 + (hr / 100) * 0.5
    if combined > 0.85:
        return "🔴 Very High"
    elif combined > 0.72:
        return "🟠 High"
    elif combined > 0.58:
        return "🟡 Moderate"
    elif combined > 0.42:
        return "🟢 Low"
    else:
        return "⚪ Recovery"


def get_sprint_label(sprint_m: float) -> str:
    if sprint_m > 100:
        return "🔴 Very High"
    elif sprint_m > 50:
        return "🟠 High"
    elif sprint_m > 20:
        return "🟡 Moderate"
    elif sprint_m > 5:
        return "🟢 Low"
    else:
        return "⚪ Minimal"


def get_accel_label(accels: float) -> str:
    if accels > 25:
        return "🔴 Very High"
    elif accels > 18:
        return "🟠 High"
    elif accels > 10:
        return "🟡 Moderate"
    elif accels > 5:
        return "🟢 Low"
    else:
        return "⚪ Minimal"


def format_score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for score."""
    filled = int((score / 100) * width)
    return "█" * filled + "░" * (width - filled) + f" {score:.0f}"


def md_day_color(md_day: str) -> str:
    colors = {
        "MD_plus1": "#6B7280",
        "MD_plus2": "#9CA3AF",
        "MD4": "#EF4444",
        "MD3": "#F97316",
        "MD2": "#EAB308",
        "MD1": "#22C55E",
    }
    return colors.get(md_day, "#FFFFFF")


def summarize_session_data(sessions_df: pd.DataFrame) -> dict:
    """Quick KPI summary for dashboard."""
    return {
        "total_sessions": sessions_df["session_id"].nunique(),
        "total_drill_occurrences": len(sessions_df),
        "unique_exercises": sessions_df["exercise_id"].nunique(),
        "avg_sprint_m": round(sessions_df["avg_sprint_distance_m"].mean(), 1),
        "avg_accelerations": round(sessions_df["avg_accelerations"].mean(), 1),
        "avg_player_load": round(sessions_df["avg_player_load"].mean(), 1),
        "avg_rpe": round(sessions_df["rpe_mean"].mean(), 2),
        "avg_hr_mean": round(sessions_df["avg_hr_mean"].mean(), 1),
    }

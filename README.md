# ⚽ Football Training Intelligence System (FTIS)

> **Data-driven drill classification · GPS load analytics · Coach query engine · Microcycle support**

---

## Overview

FTIS is a football performance engineering prototype that helps coaching and sports science staff **select the right training drill for the right training context** using structured GPS load analytics and a rules-based recommendation engine.

This is not a generic chatbot. It is a structured, data-driven intelligence engine built on football-specific physical profiling logic.

---

## Project Structure

```
ftis/
├── app.py                          # Streamlit interface (premium dark theme)
├── generate_data.py                # Synthetic dataset generator + physics profiles
├── utils.py                        # Recommendation engine + scoring logic
├── requirements.txt
├── README.md
└── data/
    ├── exercise_templates.csv          # 38 drill templates with full metadata
    ├── exercise_sessions.csv           # ~1000 simulated drill occurrences (auto-generated)
    ├── optional_player_readiness.csv   # Per-player pre-session wellness data (auto-generated)
    └── microcycle_reference.csv        # Weekly microcycle reference model
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic datasets (optional — app auto-generates on first launch)
```bash
python generate_data.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                FOOTBALL TRAINING INTELLIGENCE SYSTEM             │
├──────────────────┬─────────────────────┬────────────────────────┤
│  DATA LAYER      │  LOGIC LAYER        │  INTERFACE LAYER       │
│                  │                     │                        │
│ exercise_        │ generate_data.py    │ app.py (Streamlit)     │
│ templates.csv    │  ├─ PhysicalProfile │  ├─ Hero Dashboard     │
│                  │  ├─ MDayModifier    │  ├─ Drill Explorer     │
│ exercise_        │  ├─ FatigueModifier │  ├─ Coach Query Engine │
│ sessions.csv     │  └─ NoiseLayer      │  ├─ Drill Score Cards  │
│                  │                     │  └─ Microcycle View    │
│ player_          │ utils.py            │                        │
│ readiness.csv    │  ├─ DrillProfiler   │                        │
│                  │  ├─ Recommender     │                        │
│ microcycle_      │  ├─ GaussianScorer  │                        │
│ reference.csv    │  └─ RationaleBuilder│                        │
└──────────────────┴─────────────────────┴────────────────────────┘
```

**Flow:** Coach query → Filter by md_day / objective → Score drills against physical profile → Rank → Display top 5 with rationale

---

## Data Model

### exercise_templates.csv
38 drill templates covering: SSGs, rondos, large-sided games, sprint drills, pressing games, finishing drills, activation drills, recovery drills, possession games, transition games.

Key fields: `exercise_id`, `exercise_name`, `exercise_type`, `tactical_objective`, `physical_objective`, `pitch_length_m`, `pitch_width_m`, `area_per_player_m2`, `players_total`, `duration_min`, `sprint_profile`, `acceleration_profile`, `intensity_profile`, `recommended_md_day`

### exercise_sessions.csv
~1000 simulated drill occurrences across 150+ training sessions.

GPS metrics per row: `avg_total_distance_m`, `avg_hsr_m`, `avg_sprint_distance_m`, `avg_percent_max_speed`, `avg_accelerations`, `avg_decelerations`, `avg_player_load`, `avg_metabolic_power`, `avg_hr_mean`, `avg_hr_max`, `rpe_mean`

### optional_player_readiness.csv
Per-player pre-session wellness data: HRV, resting HR, CMJ height, VBT velocity, sleep score, fatigue/soreness/stress scores, injury flag.

### microcycle_reference.csv
6 MD-day archetypes: MD+1 (Recovery) → MD-4 (Fitness) → MD-3 (Tactical) → MD-2 (Speed) → MD-1 (Activation) with target intensity, sprint exposure, acceleration exposure, and volume benchmarks.

---

## Drill Naming Convention

```
[Type]_[Format]_[PitchSize]_[Variant]

Examples:
  SSG_5v5_30x25_Transition
  Rondo_6v2_15x15
  Flying_Sprint_30m
  LargeGame_8v8_60x40_Pressing
  PressingGame_5v5_30x25_High
  Activation_Passing_8v0
```

---

## Recommendation Engine

**Stage 1 — Scoring (5 dimensions)**

| Dimension | Weight | How computed |
|-----------|--------|--------------|
| Intensity Match | 25% | HR mean + RPE + metabolic power + dist/min → Gaussian proximity |
| Sprint Exposure | 20% | Sprint dist + max speed % + HSR → Gaussian proximity |
| Acceleration | 20% | Accel count + decel count → Gaussian proximity |
| Objective Match | 20% | Keyword matching on physical/tactical objective + type |
| MD Day Alignment | 15% | Exact match = 1.0, adjacent = 0.65, incompatible = 0.10 |

**Stage 2 — Composite Score**
`composite = Σ(dimension_score × weight) × 100`

**Stage 3 — Output**
Top 5 drills with score breakdown + natural language rationale string.

---

## Example Coach Queries

| Query | Settings |
|-------|----------|
| High intensity drill for MD-4 | Intensity: high, MD Day: MD4 |
| Speed exposure for MD-2 | Objective: speed, Sprint: high, MD Day: MD2 |
| High acceleration, low sprint | Accel: very_high, Sprint: low |
| Recovery drill for MD+1 | Recovery Mode: ON, MD Day: MD_plus1 |
| Aerobic conditioning MD-3 | Objective: aerobic, MD Day: MD3 |

---

## Data Generation Logic — Key Assumptions

1. **Physical profiles** are assigned per drill family, not random. Flying sprint drills have very high sprint distance and low accelerations; SSGs have very high accelerations and low sprint; rondos have very low everything.

2. **Area per player** modulates load: wider space = more running volume. Compact space = more accelerations.

3. **MD day multiplier**: MD4 × 1.15, MD3 × 1.02, MD2 × 0.95, MD1 × 0.68, MD+1 × 0.60.

4. **Fatigue/readiness** net modifier: `1 + (readiness-5)×0.012 - (fatigue-5)×0.009`.

5. **Noise**: ±8-16% Gaussian per metric to simulate real session-to-session variability.

6. **Metric definitions** align with commercial GPS provider standards (Catapult, STATSports, GPSports).

---

## Future Upgrades

- [ ] LLM natural language query parsing ("I want something intense but not too much running")
- [ ] ML-based drill classifier from GPS fingerprint
- [ ] Automated weekly microcycle builder (drag & drop + load constraint checking)
- [ ] Player readiness dashboard integration (real-time HRV, CMJ, wellness)
- [ ] Session load forecasting for injury risk modelling
- [ ] Real club GPS data connector (API bridge to Catapult / STATSports)
- [ ] Multi-team comparative analytics

---

## Disclaimer

This is a **prototype using synthetic data**. No real club GPS data is used. The synthetic dataset is designed to be physically realistic and defensible for demonstration and research purposes, but does not represent any specific club or player.

---

*Built as an engineering prototype for football performance intelligence. All metrics follow GPS sports science conventions.*

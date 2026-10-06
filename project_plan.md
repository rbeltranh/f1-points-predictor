# F1 Top-10 Prediction ML Project

## Objective

Predict whether an F1 driver will finish in the **top 10** of the race using information available from **FP1, FP2, and FP3**.

The prediction must simulate a real-world scenario where the model makes its prediction **after FP3 and before qualifying/race results are known**.

The project will investigate whether telemetry-derived features — including **differential-geometry features describing the relationship between speed and track curvature** — improve prediction performance.

---

# Phase 0 — Define the Experiment

* [ ] Define the exact prediction point: after FP3, before qualifying
* [ ] Define the target:

  * `1` → driver finishes P1–P10
  * `0` → driver finishes P11+
* [ ] Define what information is allowed as model input
* [ ] Define what information constitutes data leakage
* [ ] Define the initial evaluation metrics
* [ ] Define the historical period to use

---

# Phase 1 — Data Acquisition

## Data Sources

* [ ] Choose primary F1 data source
* [ ] Verify availability of FP1 telemetry
* [ ] Verify availability of FP2 telemetry
* [ ] Verify availability of FP3 telemetry
* [ ] Verify availability of race results
* [ ] Verify availability of driver/team information
* [ ] Verify availability of tire information
* [ ] Determine historical coverage

## Data Pipeline

* [ ] Create reproducible data downloader
* [ ] Download raw session data
* [ ] Save raw data without modification
* [ ] Create data directory structure
* [ ] Create data manifest
* [ ] Record unavailable/missing sessions
* [ ] Add logging/error handling to downloader

Suggested structure:

```text
data/
├── raw/
├── processed/
└── metadata/
```

---

# Phase 2 — Build the Observation Dataset

## Observation Unit

One observation should represent:

```text
one driver × one Grand Prix
```

Example:

```text
race      driver    top_10
Bahrain   VER       1
Bahrain   PER       1
Bahrain   LEC       1
Bahrain   NOR       1
```

## Tasks

* [ ] Build race-results dataset
* [ ] Create `final_position`
* [ ] Create binary `top_10` target
* [ ] Build driver/session mapping
* [ ] Join FP1 data
* [ ] Join FP2 data
* [ ] Join FP3 data
* [ ] Handle drivers missing a practice session
* [ ] Handle driver substitutions
* [ ] Document missing-data rules
* [ ] Validate number of observations per race

Target dataset concept:

```text
race
driver
season
FP1 features...
FP2 features...
FP3 features...
top_10
```

---

# Phase 3 — Lap-Level Telemetry Features

Build features for each:

```text
driver × session × lap
```

## Speed

* [ ] Mean speed
* [ ] Median speed
* [ ] Maximum speed
* [ ] Speed standard deviation
* [ ] Sector-level speed statistics

## Throttle

* [ ] Mean throttle
* [ ] Median throttle
* [ ] Full-throttle percentage
* [ ] Throttle variability
* [ ] Throttle application/reapplication metrics

## Braking

* [ ] Brake percentage
* [ ] Number of braking events
* [ ] Average braking duration
* [ ] Braking intensity
* [ ] Braking distance
* [ ] Speed at braking onset
* [ ] Minimum speed after braking

## RPM / Gear

* [ ] Mean RPM
* [ ] Maximum RPM
* [ ] RPM variability
* [ ] Mean gear
* [ ] Maximum gear
* [ ] Gear-change count

---

# Phase 4 — Session-Level Features

Aggregate lap-level features into:

```text
driver × session
```

## Pace

* [ ] Best lap time
* [ ] Median lap time
* [ ] Mean lap time
* [ ] Lap-time percentile statistics

## Consistency

* [ ] Lap-time standard deviation
* [ ] Lap-time coefficient of variation
* [ ] Fastest lap vs median lap
* [ ] Consecutive-lap consistency

## Long-Run Performance

* [ ] Identify long-run stints
* [ ] Calculate long-run average pace
* [ ] Calculate long-run pace variance
* [ ] Calculate tire degradation
* [ ] Calculate lap-time degradation slope

## Tire Information

* [ ] Tire compound
* [ ] Tire age
* [ ] Stint length
* [ ] Compound-specific pace
* [ ] Compound-specific degradation

---

# Phase 5 — Differential Geometry Features

Use the methodology from the previous F1 differential-geometry analysis as the foundation.

Reference:

https://medium.com/@rafaelbeltranhernandez/f1-data-analysis-in-python-a-differential-geometry-analysis-e59d6736c07f

## Core Feature

* [ ] Reproduce racing-line curvature calculation
* [ ] Validate curvature calculation against previous implementation
* [ ] Calculate speed-curvature correlation

## Additional Geometry Features

* [ ] Mean curvature
* [ ] Maximum curvature
* [ ] Curvature standard deviation
* [ ] High-curvature percentage
* [ ] Mean speed in high-curvature sections
* [ ] Speed distribution by curvature
* [ ] Investigate curvature-adjusted speed

## Aggregation

* [ ] Aggregate telemetry-level geometry features to lap level
* [ ] Aggregate lap-level geometry features to session level
* [ ] Validate geometry features across multiple circuits
* [ ] Investigate whether circuit layout creates systematic differences

---

# Phase 6 — FP1 → FP2 → FP3 Evolution

Create features describing how performance changes throughout the weekend.

## Pace

* [ ] FP1 → FP2 pace improvement
* [ ] FP2 → FP3 pace improvement
* [ ] FP1 → FP3 pace improvement

## Speed

* [ ] FP1 → FP2 speed improvement
* [ ] FP2 → FP3 speed improvement
* [ ] FP1 → FP3 speed improvement

## Consistency

* [ ] FP1 → FP3 consistency improvement
* [ ] Compare degradation between sessions

## Geometry

* [ ] FP1 → FP3 curvature-correlation change
* [ ] FP1 → FP3 high-curvature performance change

---

# Phase 7 — Relative Features

## Relative to Field

* [ ] Session pace percentile
* [ ] Session pace rank
* [ ] Relative mean speed
* [ ] Relative sector performance
* [ ] Relative long-run pace

## Relative to Teammate

* [ ] FP1 teammate pace delta
* [ ] FP2 teammate pace delta
* [ ] FP3 teammate pace delta
* [ ] Teammate speed delta
* [ ] Teammate long-run pace delta
* [ ] Teammate geometry-feature delta

Goal:

> Separate driver-specific performance from car-specific performance where possible.

---

# Phase 8 — Context Features

Keep these secondary to telemetry.

## Driver

* [ ] Recent finishing-position average
* [ ] Recent points average
* [ ] Recent form
* [ ] Historical performance at circuit
* [ ] Championship position before race

## Constructor

* [ ] Recent constructor performance
* [ ] Recent constructor points
* [ ] Historical circuit performance
* [ ] Championship position before race

## Important

* [ ] Verify every contextual feature existed before the prediction point
* [ ] Check for temporal leakage

---

# Phase 9 — Dataset v1

Build the first complete ML dataset.

Example:

```text
race_id
season
driver

fp1_best_lap
fp1_mean_speed
fp1_long_run_pace
fp1_speed_curvature_corr

fp2_best_lap
fp2_mean_speed
fp2_long_run_pace
fp2_speed_curvature_corr

fp3_best_lap
fp3_mean_speed
fp3_long_run_pace
fp3_speed_curvature_corr

pace_improvement_fp1_fp3
teammate_pace_delta
field_relative_pace

driver_recent_form
constructor_recent_form

top_10
```

* [ ] Generate dataset v1
* [ ] Validate schema
* [ ] Check missing values
* [ ] Check duplicates
* [ ] Check target distribution
* [ ] Save processed dataset
* [ ] Make dataset generation reproducible

---

# Phase 10 — Exploratory Data Analysis

## Target

* [ ] Calculate class balance
* [ ] Plot top-10 vs non-top-10 distribution
* [ ] Check class imbalance across seasons

## Features

* [ ] Feature distributions
* [ ] Outlier analysis
* [ ] Correlation analysis
* [ ] Missingness analysis
* [ ] Multicollinearity analysis

## F1-Specific Analysis

* [ ] Compare FP1/FP2/FP3 pace
* [ ] Compare long-run vs short-run performance
* [ ] Investigate teammate-relative features
* [ ] Investigate speed-curvature relationship
* [ ] Investigate geometry features vs race outcome

---

# Phase 11 — Baseline Models

Start simple.

## Baseline 0

* [ ] Always predict the majority class
* [ ] Record baseline metrics

## Baseline 1

* [ ] Logistic Regression
* [ ] Evaluate
* [ ] Record metrics

## Model 2

* [ ] Random Forest
* [ ] Evaluate
* [ ] Record metrics

## Model 3

* [ ] Gradient Boosting / XGBoost
* [ ] Evaluate
* [ ] Record metrics

---

# Phase 12 — Temporal Validation

**Do not use a random train/test split as the primary evaluation methodology.**

Simulate real-world prediction.

Example:

```text
2018 ───────── 2022 | 2023
        TRAIN       TEST

2018 ───────────── 2023 | 2024
          TRAIN         TEST

2018 ───────────────── 2024 | 2025
             TRAIN          TEST
```

Tasks:

* [ ] Implement chronological train/test split
* [ ] Implement rolling/expanding validation
* [ ] Evaluate baseline models
* [ ] Compare performance across seasons
* [ ] Investigate performance degradation over time

---

# Phase 13 — Feature Ablation Experiments

This is a major research component.

## Model A — Context Only

```text
Historical driver/team/context features
```

* [ ] Train
* [ ] Evaluate

## Model B — Telemetry

```text
FP1 + FP2 + FP3 telemetry
```

* [ ] Train
* [ ] Evaluate

## Model C — Telemetry + Context

```text
Telemetry + historical/context features
```

* [ ] Train
* [ ] Evaluate

## Model D — Telemetry + Context + Geometry

```text
Telemetry + context + differential geometry
```

* [ ] Train
* [ ] Evaluate

## Main Research Question

> Does differential-geometry-based feature engineering improve out-of-sample prediction of top-10 race finishes?

---

# Phase 14 — Model Interpretation

For the final model:

* [ ] Feature importance
* [ ] SHAP analysis
* [ ] Confusion matrix
* [ ] Precision
* [ ] Recall
* [ ] F1 score
* [ ] ROC-AUC
* [ ] PR-AUC
* [ ] Probability calibration

## Investigate

* [ ] Which telemetry features matter most?
* [ ] Does FP1/FP2/FP3 provide the strongest signal?
* [ ] Do long-run features matter?
* [ ] Do teammate-relative features matter?
* [ ] Does differential geometry matter?
* [ ] Which features contribute to false positives?
* [ ] Which features contribute to false negatives?

---

# Phase 15 — Model Selection

* [ ] Select final model
* [ ] Document model-selection criteria
* [ ] Document hyperparameters
* [ ] Retrain final model using appropriate historical data
* [ ] Save model artifact
* [ ] Save preprocessing pipeline
* [ ] Record experiment metadata

---

# Phase 16 — Productionization

Use the existing `rust-ml-e2e` project as the deployment blueprint.

```text
F1 data
   ↓
Feature pipeline
   ↓
Trained model
   ↓
ONNX
   ↓
Rust inference service
   ↓
AWS Lambda
   ↓
HTTP API
```

Tasks:

* [ ] Export model to ONNX
* [ ] Validate ONNX predictions against Python model
* [ ] Integrate model with Rust inference service
* [ ] Add F1-specific request schema
* [ ] Add input validation
* [ ] Deploy to AWS Lambda
* [ ] Add GitHub Actions CI/CD
* [ ] Add CloudWatch logging
* [ ] Load-test inference endpoint
* [ ] Record latency/throughput

---

# Phase 17 — Final Project Documentation

## README

* [ ] Problem definition
* [ ] Motivation
* [ ] Dataset
* [ ] Prediction point
* [ ] Data pipeline
* [ ] Feature engineering
* [ ] Differential geometry methodology
* [ ] Leakage prevention
* [ ] Temporal validation
* [ ] Models evaluated
* [ ] Results
* [ ] Ablation experiments
* [ ] Model interpretation
* [ ] Production architecture
* [ ] Performance measurements
* [ ] Limitations
* [ ] Future work

## Reproducibility

* [ ] requirements/environment file
* [ ] Setup instructions
* [ ] Data-download instructions
* [ ] Training instructions
* [ ] Evaluation instructions
* [ ] Inference instructions

---

# Final Deliverables

By the end of the project, the repository should contain:

* [ ] Reproducible data pipeline
* [ ] Clean feature-engineering pipeline
* [ ] FP1/FP2/FP3 telemetry dataset
* [ ] Differential-geometry feature pipeline
* [ ] Multiple ML models
* [ ] Temporal evaluation
* [ ] Feature ablation study
* [ ] Model interpretation
* [ ] Final trained model
* [ ] ONNX model
* [ ] Rust inference service
* [ ] AWS deployment
* [ ] CI/CD
* [ ] Performance benchmarks
* [ ] Professional README

---

# ⭐ Minimum Viable Project

If the full roadmap becomes too large, **do not compromise the scientific methodology just to add deployment features.**

The minimum strong Data Science version is:

* [ ] FP1/FP2/FP3 dataset
* [ ] Top-10 target
* [ ] Clean feature engineering
* [ ] Differential-geometry features
* [ ] Temporal train/test split
* [ ] Logistic Regression baseline
* [ ] Random Forest / Gradient Boosting
* [ ] Feature ablation
* [ ] Model interpretation
* [ ] Results documented

The AWS/Rust deployment can then be added as a second stage.

---

# 🚀 Current Priority

Only work on the **first unchecked tasks**.

### Right now:

* [ ] Choose the F1 data source
* [ ] Determine historical seasons
* [ ] Verify FP1/FP2/FP3 telemetry availability
* [ ] Verify race-result availability
* [ ] Define the raw-data schema

**Do not start modeling yet.**

The first milestone is a reliable dataset. Everything else depends on it.

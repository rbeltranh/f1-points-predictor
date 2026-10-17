# F1 Points Predictor

This portfolio project uses Formula 1 practice-session data to estimate whether a driver will finish a race in the top 10.

It demonstrates an end-to-end machine learning workflow:

1. Download historical F1 data with FastF1.
2. Turn telemetry, lap, tyre, weather, and driver information into features.
3. Add race results as the training label (`top10`).
4. Train and evaluate a scikit-learn logistic regression pipeline.
5. Track local experiments with MLflow or save model files in CI.
6. Run tests and retrain automatically with GitHub Actions.

The model is a learning and engineering demonstration, not a guarantee of race results.

## Project structure

```text
.
├── .github/workflows/
│   ├── retrain_model.yml          # Retrain and publish model releases
│   └── run_tests_on_push.yml      # Run unit tests on pushes and pull requests
├── src/
│   ├── eda/
│   │   └── exploratory_data_analysis.ipynb
│   ├── feature_processing/
│   │   ├── feature_functions.py  # Feature engineering from FastF1 sessions
│   │   ├── get_dataset.py        # Download, label, checkpoint, and combine data
│   │   └── all_data.csv          # Combined training dataset
│   └── model_training/
│       ├── train_model.py         # Training, evaluation, and model export
│       └── artifacts/             # MLflow model artifacts and CI outputs
├── test/test_features.py          # Unit tests using mocked FastF1 objects
├── Makefile                       # Common commands
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
└── mlflow.db                      # Local MLflow tracking database
```

The FastF1 cache and per-race checkpoints are generated locally and ignored by Git. This makes an interrupted download resumable without adding a large cache to the repository.

## Data engineering workflow

The pipeline is in `src/feature_processing` and uses FastF1 as its source.

### 1. Select races

`get_dataset.py` loops over seasons 2022 through 2026. It reads each event schedule and keeps races whose fifth session date is at least four days in the past. This avoids training on a race weekend that is still being completed.

FastF1 caching is enabled in `src/feature_processing/cache`. Each completed Grand Prix is also written to `src/feature_processing/checkpoints/<season>_<round>.csv`.

### 2. Build features

For each race, `feature_functions.py` loads FP1, FP2, and FP3. For every driver with usable data, it takes the fastest lap and creates:

- Context: race, season, driver, and team.
- Lap performance: lap time and the three sector times.
- Tyres: compound and tyre age.
- Speed, engine, throttle, and gear summaries.
- Telemetry relationships: RPM-to-speed and throttle-to-speed correlations.
- Track-shape information: mean and maximum curvature, plus the relationship between speed and curvature.
- Session-wide conditions: fastest lap and minimum, mean, and maximum air and track temperatures.

The curvature calculation uses telemetry `X` and `Y` coordinates to estimate how sharply the car is turning during the lap.

### 3. Create the target label

The script separately loads race results and creates `top10`:

- `1` when a driver finishes in position 10 or better.
- `0` otherwise.

Practice features and race labels are joined using season, race name, and driver number. A checkpoint is written only after both features and race results for a Grand Prix are available.

### 4. Combine the data

After each run, all available checkpoints are combined into `src/feature_processing/all_data.csv`. A rate-limited or interrupted run therefore keeps the work already completed.

FastF1 can return a rate-limit error. The `create-dataset` Make target recognizes exit code `75`, waits 20 minutes, and retries. Other errors stop the command so they can be investigated.

## Model training (`src/model_training`)

`train_model.py` reads `src/feature_processing/all_data.csv` and fails early with a helpful message if the dataset does not exist.

Before training, it:

1. Separates `top10` from the input columns.
2. Removes rows containing missing values or infinite numeric values.
3. Splits the data into training and test sets with a fixed seed (`44`) and a 15% test size.
4. Scales numeric columns with `MinMaxScaler`.
5. One-hot encodes text columns such as team, tyre compound, race, driver, and session.
6. Fits a `LogisticRegression` classifier inside one scikit-learn `Pipeline`.

The three differential-geometry features receive a transformer weight of `3.0`, other numeric features `1.0`, and categorical features `0.5`. This makes the telemetry track-shape signals more influential in the pipeline.

The script reports test accuracy and creates a feature-importance table from the logistic regression coefficients. It also records the train/test sizes, classifier, transformer weights, timestamp, scikit-learn version, and (in GitHub Actions) the commit SHA.

### Local training and MLflow

By default, a local run logs to MLflow:

```bash
make train-model
```

To open the local MLflow tracking server:

```bash
make run-mlfow-server
```

The server uses `mlflow.db` as its backend and stores model artifacts under `src/model_training/artifacts`. The training script uses the `f1-top10-training` experiment, logs accuracy and parameters, saves feature importance, and registers the model as `f1-top10-predictor`.

The committed MLflow artifacts include the serialized scikit-learn model, environment information, input examples, and model signature, making the trained pipeline inspectable and reproducible.

### Training in CI

GitHub Actions sets `GITHUB_ACTIONS=true`. The script detects this and skips the MLflow server, writing these files to `src/model_training/artifacts/remote_workflows`:

- `model_pipeline.joblib`
- `metrics.json`
- `feature_importance.csv`

To exercise this mode locally, set `TRAIN_MODE=remote`. `TRAIN_MODE=local` forces MLflow mode when needed.

## GitHub Actions: testing, retraining, and releases

### Continuous integration: `run_tests_on_push.yml`

This workflow runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. It checks out the repository, installs Python 3.14.7 with `uv`, installs the exact locked dependencies, and runs `make run-unit-tests`.

The tests use mocked FastF1 sessions, so they check feature calculations without network requests or a FastF1 cache. They cover curvature and driver/session feature generation.

### Retraining: `retrain_model.yml`

This workflow starts when `src/feature_processing/all_data.csv` changes on `main`, or when manually dispatched. In practice, data generation is run locally (or by another process), the updated CSV is committed, and that commit starts retraining.

The workflow installs locked dependencies and trains in CI mode. It then:

1. Prints the new accuracy.
2. Uploads the model, metrics, and feature-importance files as a workflow artifact.
3. Downloads metrics from the latest GitHub release, when one exists.
4. Compares new accuracy with the latest released model.
5. Publishes a new GitHub release containing the model files only when accuracy is at least as good as the previous release.

Each release includes the accuracy and source commit. Concurrency cancellation means a newer data update replaces an older retraining run that has not finished.

This gives the project a simple model-promotion rule: every candidate remains available as a CI artifact, while only a model that passes the accuracy comparison becomes a release.

## Setup

The project uses `uv` for Python and dependency management. Python 3.14 or newer is declared in `pyproject.toml`; CI pins Python 3.14.7.

```bash
uv sync
```

Generate or update the dataset:

```bash
make create-dataset
```

For a long download on macOS, run it in the background and follow its log:

```bash
make create-dataset-bg
make dataset-log
```

Run unit tests:

```bash
make run-unit-tests
```

The EDA notebook at `src/eda/exploratory_data_analysis.ipynb` can be opened in Jupyter after the dataset has been generated. It gives a hands-on view of the combined data before training.

## Make targets

| Command | Purpose |
| --- | --- |
| `make create-dataset` | Download data, build checkpoints, and write `all_data.csv` |
| `make create-dataset-bg` | Run dataset creation in the background on macOS |
| `make dataset-log` | Follow the background download log |
| `make run-unit-tests` | Run the feature-generation unit tests |
| `make train-model` | Train the model and log it to MLflow locally |
| `make run-mlfow-server` | Start the local MLflow UI and tracking server |

## Limitations and possible next steps

This is a portfolio baseline with clear directions for improvement:

- Use a time-based or race-based validation split to better represent predicting a future race.
- Add precision, recall, ROC-AUC, and calibration metrics alongside accuracy.
- Tune the classification threshold and compare additional models.
- Check carefully which practice-session information would really be available before a race to reduce leakage risk.
- Add a prediction interface or dashboard for an upcoming race.
- Move dataset generation and model artifacts to dedicated storage as the project grows.

The repository shows a complete, understandable baseline that can be improved in measurable steps.

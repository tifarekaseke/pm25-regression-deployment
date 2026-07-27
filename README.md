# PM2.5 Air-Quality Regression and Mobile Prediction App

## Mission and problem
My mission is to support healthier urban communities through accessible environmental information. This project predicts hourly PM2.5 concentration so that users can estimate air-pollution conditions from pollutant, weather, time, wind, and monitoring-station data. The output is a continuous value measured in µg/m³, making this a regression problem. The prediction is delivered through a FastAPI service and a one-page Flutter mobile application.

## Dataset
**Source:** UCI Machine Learning Repository — Beijing Multi-Site Air Quality dataset: https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata

The dataset contains 420,768 hourly observations from 12 monitoring stations between March 2013 and February 2017. It includes six air pollutants, six meteorological variables, time fields, wind direction, station name, and missing values. The target is `PM2.5` concentration in µg/m³.

Video Link :(https://youtu.be/vhXHyZ5CqB4)
## Repository structure

```text
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb
│   │   └── training.py
│   ├── API/
│   │   └── prediction.py
│   └── FlutterApp/
│       ├── lib/main.dart
│       └── pubspec.yaml
├── data/
├── models/
├── pyproject.toml
├── uv.lock                         # generate with `uv lock`
├── requirements.txt
├── render.yaml
└── README.md
```

## Features used

| API field | Dataset column | Type | Validation |
|---|---|---:|---|
| month | month | integer | 1–12 |
| hour | hour | integer | 0–23 |
| pm10 | PM10 | float | 0–1000 |
| so2 | SO2 | float | 0–500 |
| no2 | NO2 | float | 0–500 |
| co | CO | float | 0–10000 |
| o3 | O3 | float | 0–500 |
| temperature | TEMP | float | −40–50 °C |
| pressure | PRES | float | 900–1100 hPa |
| dew_point | DEWP | float | −50–40 °C |
| rainfall | RAIN | float | 0–100 mm |
| wind_speed | WSPM | float | 0–50 m/s |
| wind_direction | wd | string | one of the 16 compass directions |
| station | station | string | one of the 12 dataset stations |

### Feature engineering decisions

- `No` is dropped because it is only a row identifier and has no predictive meaning.
- `year` and `day` are dropped to reduce dependence on specific historical dates; `month` and `hour` retain seasonal and daily patterns.
- Rows with a missing target are removed because supervised training requires a known PM2.5 value.
- Missing numeric predictor values are replaced with the training median.
- Missing categorical values are replaced with the most frequent training category.
- `wd` and `station` are converted to numeric one-hot columns.
- Numeric values are standardized. This is essential for both gradient-descent linear regressions because variables such as CO, pressure, and rainfall have very different scales.

## Models and evaluation

The notebook compares exactly four algorithms required by the updated rubric:

1. **Batch Gradient Descent Linear Regression** — a custom scikit-learn-compatible estimator that updates coefficients using the full training batch.
2. **Stochastic Gradient Descent Linear Regression (`SGDRegressor`)** — a second linear model trained through stochastic gradient descent.
3. **Decision Tree Regressor** — the standalone tree algorithm.
4. **Random Forest Regressor** — the ensemble algorithm, combining many decision trees.

The models are evaluated using MSE, RMSE, MAE, and R². The model with the lowest test RMSE is saved to `models/best_model.joblib`, and its name, features, and metrics are saved to `models/model_metadata.json`.

The notebook also contains:

- Target distribution and missing-value visualizations
- Correlation heatmap
- PM10 versus PM2.5 scatter plot
- Separate train and test loss curves for Batch GD and SGD
- Before/after scatter plots showing a fitted linear line
- Model-comparison chart
- A prediction made from one row of the test set

## Python setup with uv

Install uv, then create the lock file and environment:

```bash
uv lock
uv sync
```

Open the notebook:

```bash
uv run jupyter lab summative/linear_regression/multivariate.ipynb
```

Run all cells. The notebook downloads the public dataset, uses a reproducible 120,000-row modelling sample to keep training and the saved forest practical, trains all four models, and creates the saved model required by the API.

## Run the API locally

After generating the model:

```bash
uv run uvicorn summative.API.prediction:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example request to `POST /predict`:

```json
{
  "month": 7,
  "hour": 14,
  "pm10": 120,
  "so2": 12,
  "no2": 45,
  "co": 900,
  "o3": 70,
  "temperature": 28.5,
  "pressure": 1004,
  "dew_point": 17.2,
  "rainfall": 0,
  "wind_speed": 2.4,
  "wind_direction": "SE",
  "station": "Aotizhongxin"
}
```

## CORS configuration and reasoning

The API does **not** use `allow_origins=["*"]`. Allowed origins are read from the `ALLOWED_ORIGINS` environment variable and default to known local-development addresses. The deployed frontend origin should be added to this list.

- Allowed methods: `GET`, `POST`, and `OPTIONS` because the service only checks health, predicts, and accepts retraining uploads.
- Restricted methods: `PUT`, `PATCH`, and `DELETE` because they are not required.
- Allowed headers: `Content-Type`, `Authorization`, and `X-Retrain-Token`.
- Credentials are disabled because the mobile prediction request does not use cookies.
- A native Flutter mobile app is not governed by browser CORS in the same way as Flutter Web, but explicit CORS remains useful for Swagger/browser clients and meets the security requirement.

## Retraining with new data

`POST /retrain` accepts a CSV containing all 14 feature columns and the `PM2.5` target. It validates the uploaded columns, retrains all four models, saves the new best model, reloads it in the API process, and returns the new metrics.

Set a secret token:

```bash
export RETRAIN_TOKEN="replace-with-a-strong-secret"
```

In Swagger UI, send the same value in the `X-Retrain-Token` header. The file upload endpoint is intentionally protected so that unknown users cannot trigger expensive retraining.

## Deploy on Render

1. Run the notebook and commit `models/best_model.joblib` and `models/model_metadata.json` to GitHub. They are ignored by default, so force-add them once:

```bash
git add -f models/best_model.joblib models/model_metadata.json
```

2. Push the repository to GitHub.
3. Create a Render **Web Service** from the repository or use `render.yaml` as a Blueprint.
4. Confirm these commands:

```text
Build: pip install uv && uv sync --no-dev
Start: uv run uvicorn summative.API.prediction:app --host 0.0.0.0 --port $PORT
```

5. Add the `ALLOWED_ORIGINS` and `RETRAIN_TOKEN` environment variables.
6. Insert the deployed URL below:

```text
Public Swagger URL: https://YOUR-SERVICE-NAME.onrender.com/docs
Prediction endpoint: https://YOUR-SERVICE-NAME.onrender.com/predict
```

## Flutter mobile app

From `summative/FlutterApp`, create any missing platform folders once and install dependencies:

```bash
flutter create --project-name air_quality_predictor .
flutter pub get
```

If Flutter replaces `lib/main.dart` or `pubspec.yaml`, restore those two files from Git. In `lib/main.dart`, replace:

```dart
https://YOUR-SERVICE-NAME.onrender.com/predict
```

with the Render prediction URL. Then run:

```bash
flutter run
```

The app provides 14 organized inputs, a **Predict** button, loading feedback, the predicted PM2.5 value, and readable validation/API errors.



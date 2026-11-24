"""
Train the final model for the demo and save it to preprocessor.joblib and xgb_model.json.

Uses XGBoost with the best settings found by grid search in experiment.py,
trained on exactly the same 80% training split (same seed, same duplicate removal),
so the saved model matches the model evaluated in experiment.py.

The XGBoost model is saved with save_model (JSON) instead of being pickled, because
a pickled XGBoost model contains a random-number state whose format depends on the
operating system (a model pickled on Linux failed to load on Windows with
"input stream corrupted"). The JSON file has no such state and loads on any OS.

How to run:
    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from experiment import CATEGORICAL, DATA_PATH, NUMERIC, SEED, TARGET, TEST_SIZE, make_preprocessor, rmse

# Best XGBoost settings from experiment.py (results/best_params.json)
BEST_PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8}


def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=False)),
            ("model", XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror", **BEST_PARAMS)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Test RMSE ${rmse(y_test, pred):,.0f} | MAE ${mean_absolute_error(y_test, pred):,.0f} | R2 {r2_score(y_test, pred):.3f}")

    joblib.dump(model.named_steps["prep"], "preprocessor.joblib")
    model.named_steps["model"].save_model("xgb_model.json")
    print("Saved preprocessor.joblib and xgb_model.json")


if __name__ == "__main__":
    main()

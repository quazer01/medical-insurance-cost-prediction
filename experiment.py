"""
CSC340 Artificial Intelligence - Project Experiment
Predicting Medical Insurance Costs with Machine Learning

Research question:
    Can machine learning models predict an individual's medical insurance
    charges more accurately than traditional methods?

What this script does:
    1. Loads the Medical Cost Personal Datasets (insurance.csv) and removes duplicate rows.
    2. Splits the data into 80% train and 20% test (fixed random seed).
    3. Evaluates 2 traditional methods and 3 AI models with the same 10-fold
       cross-validation splits on the training set:
         Traditional: group-average pricing, multiple linear regression
         AI:          Random Forest, XGBoost, neural network (MLP)
       The AI models are tuned with grid search inside 10-fold cross-validation.
    4. Tests every method once on the held-out test set (RMSE, MAE, R2).
    5. Explains the best AI model (chosen by cross-validation RMSE) with SHAP values.
    6. Saves tables and charts to the results/ folder.

How to run:
    1. Download insurance.csv from https://www.kaggle.com/datasets/mirichoi0218/insurance
       and put it in the same folder as this script.
    2. pip install pandas numpy scikit-learn xgboost shap matplotlib
    3. python experiment.py
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_PATH = "insurance.csv"
OUT_DIR = "results"
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 10

NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
TARGET = "charges"

# Age groups for group-average pricing (10-year groups; data ages are 18-64)
AGE_BINS = [17, 29, 39, 49, 59, 64]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-64"]


# ---------------------------------------------------------------------------
# Traditional method 1: group-average pricing
# ---------------------------------------------------------------------------
class GroupAveragePricing(BaseEstimator, RegressorMixin):
    """Predicts the average training charges of people in the same
    10-year age group and with the same smoking status, like an insurance
    rate table. Falls back to the overall average for an unseen group."""

    def fit(self, X, y):
        groups = self._groups(X)
        self.table_ = pd.Series(np.asarray(y), index=groups.index).groupby(groups).mean()
        self.overall_ = float(np.mean(y))
        return self

    def predict(self, X):
        groups = self._groups(X)
        return groups.map(self.table_).fillna(self.overall_).to_numpy(dtype=float)

    @staticmethod
    def _groups(X):
        age_group = pd.cut(X["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        return age_group + "|" + X["smoker"].astype(str)


# ---------------------------------------------------------------------------
# Preprocessing: one-hot encode categories; optionally scale numbers (for MLP)
# ---------------------------------------------------------------------------
def make_preprocessor(scale_numeric):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, NUMERIC),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()

    # -----------------------------------------------------------------------
    # 1. Load and clean the data
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} rows, removed {n_raw - len(df)} duplicate row(s): {len(df)} rows left.")
    print(f"Missing values: {int(df.isna().sum().sum())}")

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    # -----------------------------------------------------------------------
    # 2. Train/test split (the test set is only used once, at the end)
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Same 10 folds for every method, so the comparison is fair
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # -----------------------------------------------------------------------
    # 3. Define the 5 methods
    # -----------------------------------------------------------------------
    traditional = {
        "Group-average pricing": GroupAveragePricing(),
        "Linear regression": Pipeline(
            [("prep", make_preprocessor(scale_numeric=False)), ("model", LinearRegression())]
        ),
    }

    ai_models = {
        "Random Forest": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale_numeric=False)),
                    ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "XGBoost": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale_numeric=False)),
                    ("model", XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror")),
                ]
            ),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        ),
        # The target (charges) is also standardised for the MLP, because
        # neural networks train poorly on targets in the tens of thousands.
        # Predictions are converted back to dollars automatically.
        "Neural network (MLP)": (
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("prep", make_preprocessor(scale_numeric=True)),
                        (
                            "model",
                            MLPRegressor(
                                solver="adam",
                                max_iter=3000,
                                early_stopping=True,
                                n_iter_no_change=30,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            ),
            {
                "regressor__model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "regressor__model__alpha": [1e-4, 1e-2, 1.0],
                "regressor__model__learning_rate_init": [1e-3, 1e-2],
            },
        ),
    }

    # -----------------------------------------------------------------------
    # 4. Tune the AI models with grid search + 10-fold CV (train set only)
    # -----------------------------------------------------------------------
    final_models = dict(traditional)
    best_params = {}
    for name, (estimator, grid) in ai_models.items():
        t0 = time.time()
        search = GridSearchCV(
            estimator, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)
        final_models[name] = search.best_estimator_
        best_params[name] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        print(f"Tuned {name} in {time.time() - t0:.0f}s: {best_params[name]}")

    # -----------------------------------------------------------------------
    # 5. Cross-validation scores of all 5 methods on the same folds
    #    (for the AI models these use the tuned settings)
    # -----------------------------------------------------------------------
    cv_rows = []
    for name, model in final_models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )
        cv_rows.append(
            {
                "Method": name,
                "Type": "Traditional" if name in traditional else "AI",
                "CV RMSE mean": -scores["test_rmse"].mean(),
                "CV RMSE std": scores["test_rmse"].std(),
                "CV MAE mean": -scores["test_mae"].mean(),
                "CV R2 mean": scores["test_r2"].mean(),
            }
        )
    cv_table = pd.DataFrame(cv_rows).sort_values("CV RMSE mean").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 6. Final test: fit on the full train set, test once on the test set
    # -----------------------------------------------------------------------
    test_rows, test_preds = [], {}
    for name, model in final_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        test_preds[name] = pred
        test_rows.append(
            {
                "Method": name,
                "Type": "Traditional" if name in traditional else "AI",
                "Test RMSE": rmse(y_test, pred),
                "Test MAE": mean_absolute_error(y_test, pred),
                "Test R2": r2_score(y_test, pred),
            }
        )
    test_table = pd.DataFrame(test_rows).sort_values("Test RMSE").reset_index(drop=True)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")
    print("\n=== 10-fold cross-validation on the train set ===")
    print(cv_table.to_string(index=False))
    print("\n=== Final test set ===")
    print(test_table.to_string(index=False))

    cv_table.to_csv(f"{OUT_DIR}/cv_results.csv", index=False)
    test_table.to_csv(f"{OUT_DIR}/test_results.csv", index=False)
    with open(f"{OUT_DIR}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # 7. Charts
    # -----------------------------------------------------------------------
    colors = ["#0E9F9A" if t == "AI" else "#9AA5B1" for t in test_table["Type"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(test_table["Method"][::-1], test_table["Test RMSE"][::-1], color=colors[::-1])
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in test_table["Test RMSE"][::-1]], padding=4)
    ax.set_xlabel("Test RMSE (USD, lower is better)")
    ax.set_title("Test RMSE by method (teal = AI, grey = traditional)")
    ax.set_xlim(0, test_table["Test RMSE"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/rmse_comparison.png", dpi=150)
    plt.close(fig)

    order = list(test_table["Method"])
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4), sharex=True, sharey=True)
    lim = max(y_test.max(), max(p.max() for p in test_preds.values())) * 1.05
    for ax, name in zip(axes, order):
        ax.scatter(y_test, test_preds[name], s=10, alpha=0.6, color="#0E9F9A")
        ax.plot([0, lim], [0, lim], color="#E5484D", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Actual charges (USD)")
    axes[0].set_ylabel("Predicted charges (USD)")
    fig.suptitle("Predicted vs actual charges on the test set (red line = perfect prediction)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 8. SHAP: explain the best AI model, chosen by CV RMSE (not by test)
    # -----------------------------------------------------------------------
    ai_cv = cv_table[cv_table["Type"] == "AI"]
    best_ai = ai_cv.iloc[0]["Method"]
    model = final_models[best_ai]
    pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    prep = pipe.named_steps["prep"]
    feature_names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    X_test_t = pd.DataFrame(prep.transform(X_test), columns=feature_names)

    if best_ai in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        shap_values = explainer(X_test_t)
    else:
        X_train_t = pd.DataFrame(prep.transform(X_train), columns=feature_names)
        background = shap.sample(X_train_t, 100, random_state=SEED)

        def predict_dollars(data):
            # MLP predicts standardised charges; convert back to USD
            scaled = pipe.named_steps["model"].predict(np.asarray(data))
            return model.transformer_.inverse_transform(scaled.reshape(-1, 1)).ravel()

        explainer = shap.KernelExplainer(predict_dollars, background)
        shap_values = shap.Explanation(
            explainer.shap_values(X_test_t, nsamples=200), data=X_test_t.values, feature_names=feature_names
        )

    importance = (
        pd.DataFrame({"feature": feature_names, "mean |SHAP|": np.abs(shap_values.values).mean(axis=0)})
        .sort_values("mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(f"{OUT_DIR}/shap_importance.csv", index=False)
    print(f"\n=== SHAP feature importance ({best_ai}) ===")
    print(importance.to_string(index=False))

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=len(feature_names))
    plt.title(f"SHAP values: {best_ai}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # 9. Answer the research question
    # -----------------------------------------------------------------------
    best_trad = test_table[test_table["Type"] == "Traditional"].iloc[0]
    best_ai_test = test_table[test_table["Method"] == best_ai].iloc[0]
    gain = (best_trad["Test RMSE"] - best_ai_test["Test RMSE"]) / best_trad["Test RMSE"] * 100
    print("\n=== Research question ===")
    print(
        f"Best AI model (chosen by CV): {best_ai}, test RMSE ${best_ai_test['Test RMSE']:,.0f}\n"
        f"Best traditional method: {best_trad['Method']}, test RMSE ${best_trad['Test RMSE']:,.0f}\n"
        f"Difference: {gain:+.1f}% "
        + ("(AI is more accurate)" if gain > 0 else "(AI is NOT more accurate)")
    )
    print(f"\nDone in {time.time() - start:.0f}s. Results saved to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()

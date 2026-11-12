"""Candidate models. Every preprocessing step lives inside the estimator pipeline, so fitting on
one period and scoring another cannot leak statistics across the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, NEIGHBOURS, RAIN, SEASON, SITE_HISTORY

# Skewed non-negative columns get a log1p before scaling in the linear model.
LOG_COLS = ["r1", "r2", "r3", "r7", "r14", "r30", "max3", "n_prev", "gap_days"]
LINEAR_NUM = [c for c in RAIN + SITE_HISTORY + NEIGHBOURS + SEASON if c not in LOG_COLS]


class SiteRatePrior(BaseEstimator, ClassifierMixin):
    """Predicts each site's as-of exceedance rate and ignores the weather completely."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(X["exp_rate"].to_numpy(dtype=float), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])


class RainRule(BaseEstimator, ClassifierMixin):
    """The usual operating rule: flag a day when recent rain passes a threshold.

    The two probabilities are the observed exceedance rates on flagged and unflagged training days.
    """

    def __init__(self, threshold_mm: float = 10.0, column: str = "r3"):
        self.threshold_mm = threshold_mm
        self.column = column

    def fit(self, X, y):
        flag = X[self.column].to_numpy() >= self.threshold_mm
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.p_on_ = (y[flag].sum() + 0.5) / (flag.sum() + 1.0)
        self.p_off_ = (y[~flag].sum() + 0.5) / ((~flag).sum() + 1.0)
        return self

    def flag(self, X) -> np.ndarray:
        return X[self.column].to_numpy() >= self.threshold_mm

    def predict_proba(self, X):
        p = np.where(self.flag(X), self.p_on_, self.p_off_)
        return np.column_stack([1 - p, p])


def make_logistic(c: float = 0.1) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "log",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                LOG_COLS,
            ),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LINEAR_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=2000))])


def _tree_pre(extra: tuple[str, ...] = ()) -> ColumnTransformer:
    numeric = [c for c in FEATURES if c not in CATEGORICAL] + list(extra)
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                CATEGORICAL,
            ),
        ]
    )


def make_forest(seed: int = 7) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                [c for c in FEATURES if c not in CATEGORICAL],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=40,
        max_features=0.4,
        max_samples=0.5,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", rf)])


def make_boosting(
    max_iter: int = 200, leaves: int = 15, seed: int = 7, extra: tuple[str, ...] = ()
) -> Pipeline:
    n_num = len([c for c in FEATURES if c not in CATEGORICAL]) + len(extra)
    mask = [False] * n_num + [True] * len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=leaves,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=mask,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("pre", _tree_pre(extra)), ("clf", clf)])


# Order matters for the one-standard-error rule: lower rank means simpler.
COMPLEXITY = {"site prior": 0, "rain rule": 1, "logistic": 2, "forest": 3, "boosting": 4}


def proba(model, X: pd.DataFrame, extra: tuple[str, ...] = ()) -> np.ndarray:
    return model.predict_proba(X[FEATURES + list(extra)])[:, 1]

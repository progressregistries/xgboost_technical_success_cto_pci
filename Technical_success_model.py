"""Generic technical-success prediction pipeline.

This file is a cleaned, GitHub-ready version of the full model-development
workflow:

1. Load a source CSV through a generic column map.
2. Recode and type candidate predictors.
3. Split into training and held-out test sets.
4. Exclude high-missingness predictors using training data only.
5. Apply construct-aware de-duplication.
6. Rank predictors by bootstrap permutation importance.
7. Select a 5-variable candidate set using LASSO plus optional Boruta.
8. Optionally run model-specific one-for-one swap search.
9. Tune and compare model families.
10. Calibrate the representative model and export it.
11. Compare with J-CTO, PROGRESS-CTO, and CASTLE .
12. Run held-out fairness metrics and optional leave-one-center-out validation.

Before use, replace every ``TODO_*`` value in CONFIG with the column names from
your local dataset. The internal feature names are intentionally generic.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

try:
    from boruta import BorutaPy
except ImportError:  # pragma: no cover - optional dependency
    BorutaPy = None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FeatureSpec:
    """A candidate predictor with a public internal name and private source column."""

    name: str
    source_column: str
    kind: str | None = None  # "continuous", "binary", "categorical", or None to infer.
    yes_no_unknown: bool = False  # For source coding 1=yes, 2=no, 3=unknown.


@dataclass
class ScoreColumns:
    """Optional source columns for comparator scores and subgroup analyses."""

    age_years: str | None = "TODO_AGE_COLUMN"
    sex: str | None = "TODO_SEX_COLUMN"
    center_id: str | None = "TODO_CENTER_COLUMN"
    jcto_stump: str | None = "TODO_JCTO_STUMP_COLUMN"
    jcto_calcification: str | None = "TODO_JCTO_CALCIFICATION_COLUMN"
    jcto_tortuosity: str | None = "TODO_JCTO_TORTUOSITY_COLUMN"
    jcto_length: str | None = "TODO_JCTO_LENGTH_COLUMN"
    jcto_prior_attempt: str | None = "TODO_JCTO_PRIOR_ATTEMPT_COLUMN"
    progress_proximal_cap: str | None = "TODO_PROGRESS_PROXIMAL_CAP_COLUMN"
    progress_collateral: str | None = "TODO_PROGRESS_COLLATERAL_COLUMN"
    progress_tortuosity: str | None = "TODO_PROGRESS_TORTUOSITY_COLUMN"
    progress_circumflex: str | None = "TODO_PROGRESS_CIRCUMFLEX_COLUMN"
    castle_prior_bypass: str | None = "TODO_CASTLE_PRIOR_BYPASS_COLUMN"
    castle_age_years: str | None = "TODO_CASTLE_AGE_COLUMN"
    castle_stump_adverse: str | None = "TODO_CASTLE_STUMP_COLUMN"
    castle_tortuosity: str | None = "TODO_CASTLE_TORTUOSITY_COLUMN"
    castle_length_mm: str | None = "TODO_CASTLE_LENGTH_MM_COLUMN"
    castle_calcification: str | None = "TODO_CASTLE_CALCIFICATION_COLUMN"


@dataclass
class PipelineConfig:
    data_path: Path = Path("technical_success_input.csv")
    output_dir: Path = Path("Technical_success_model_outputs")
    outcome_column: str = "TODO_OUTCOME_COLUMN"
    positive_outcome_is_success: bool = True
    random_state: int = 42
    test_size: float = 0.20
    cv_splits: int = 5
    missingness_threshold: float = 0.40
    final_feature_count: int = 5
    use_optuna: bool = True
    optuna_trials_5var: int = 200
    model_swap_cycles: int = 360
    run_model_specific_swap_search: bool = True
    run_loco: bool = False
    n_boot: int = 2000
    candidate_features: list[FeatureSpec] = field(
        default_factory=lambda: [
            # Replace this short example list with all eligible pre-procedural
            # candidate predictors. Keep the left-side ``name`` generic.
            FeatureSpec("stump_adverse_binary", "TODO_BLUNT_OR_NO_STUMP_COLUMN", "binary"),
            FeatureSpec("lesion_length_mm", "TODO_LENGTH_MM_COLUMN", "continuous"),
            FeatureSpec("bend_gt45_binary", "TODO_BEND_GT45_COLUMN", "binary"),
            FeatureSpec("calcification_category", "TODO_CALCIFICATION_COLUMN", "categorical"),
            FeatureSpec("distal_landing_zone_good_binary", "TODO_DISTAL_LANDING_ZONE_COLUMN", "binary"),
            FeatureSpec("candidate_006", "TODO_CANDIDATE_006_COLUMN"),
            FeatureSpec("candidate_007", "TODO_CANDIDATE_007_COLUMN"),
            FeatureSpec("candidate_008", "TODO_CANDIDATE_008_COLUMN"),
        ]
    )
    leakage_or_excluded_features: set[str] = field(
        default_factory=lambda: {
            # Add generic internal names for variables that are intra-procedural,
            # post-procedural, identifiers, or otherwise not bedside predictors.
        }
    )
    construct_groups: list[list[str]] = field(
        default_factory=lambda: [
            # Add groups of interchangeable predictors; the least-missing
            # representative is kept before ranking.
            ["stump_adverse_binary", "candidate_006"],
            ["lesion_length_mm", "candidate_007"],
            ["bend_gt45_binary", "candidate_008"],
        ]
    )
    score_columns: ScoreColumns = field(default_factory=ScoreColumns)


CONFIG = PipelineConfig()


# =============================================================================
# Preprocessing helpers
# =============================================================================


class CategoricalStringNormalizer(BaseEstimator, TransformerMixin):
    """Convert mixed categorical values to strings without losing missingness."""

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            array = np.asarray(X, dtype=object)
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(array.shape[1])], dtype=object
            )
        return self

    @staticmethod
    def _normalize_value(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and float(value).is_integer():
                return str(int(value))
            return str(float(value))
        text = str(value).strip()
        return text if text else np.nan

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(np.asarray(X, dtype=object), columns=self.feature_names_in_)
        for column in frame.columns:
            frame[column] = frame[column].map(self._normalize_value)
        return frame.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


def _intset(series: pd.Series) -> set[int] | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() < 0.5:
        return None
    values = numeric.dropna().unique()
    if len(values) == 0:
        return None
    if not np.all(np.isclose(np.mod(values, 1.0), 0.0, atol=1e-6)):
        return None
    return set(np.round(values).astype(int))


def infer_kind(series: pd.Series, max_categorical_levels: int = 20) -> str:
    non_missing = series.dropna()
    n_unique = non_missing.nunique()
    if n_unique <= 1:
        return "drop"
    integer_set = _intset(series)
    numeric_fraction = pd.to_numeric(series, errors="coerce").notna().mean()
    if n_unique == 2:
        return "binary"
    if integer_set is not None and integer_set <= {1, 2, 3} and {1, 2} <= integer_set:
        return "binary"
    if numeric_fraction >= 0.5 and integer_set is not None and len(integer_set) < max_categorical_levels:
        return "categorical"
    if numeric_fraction >= 0.5:
        return "continuous"
    if n_unique < max_categorical_levels:
        return "categorical"
    return "drop"


def recode_binary(series: pd.Series, yes_no_unknown: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    integer_set = _intset(series)
    if yes_no_unknown:
        return numeric.map({1: 1.0, 2: 0.0, 3: np.nan})
    if integer_set == {1, 2}:
        return numeric.map({1: 1.0, 2: 0.0})
    if integer_set == {1, 2, 3}:
        return numeric.map({1: 1.0, 2: 0.0, 3: np.nan})
    if integer_set == {0, 1}:
        return numeric.astype(float)
    normalized = series.map(lambda value: pd.NA if pd.isna(value) else str(value).strip())
    levels = normalized.dropna().unique().tolist()
    if len(levels) != 2:
        raise ValueError(f"Binary feature has {len(levels)} non-missing levels.")
    level_map = {level: float(index) for index, level in enumerate(sorted(levels))}
    return normalized.map(level_map).astype(float)


def map_outcome(series: pd.Series, positive_is_success: bool = True) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        y = pd.to_numeric(series, errors="coerce")
    else:
        normalized = series.astype("string").str.strip().str.lower()
        y = normalized.map(
            {
                "yes": 1,
                "success": 1,
                "successful": 1,
                "true": 1,
                "1": 1,
                "no": 0,
                "failure": 0,
                "failed": 0,
                "false": 0,
                "0": 0,
            }
        )
    if not positive_is_success:
        y = 1 - y
    valid = y.isin([0, 1]) | y.isna()
    if not valid.all():
        raise ValueError("Outcome must map to 0/1.")
    return y


def check_placeholders(config: PipelineConfig) -> None:
    values = [config.outcome_column, str(config.data_path)]
    values.extend(spec.source_column for spec in config.candidate_features)
    values.extend(v for v in vars(config.score_columns).values() if isinstance(v, str))
    todos = [v for v in values if isinstance(v, str) and v.startswith("TODO_")]
    if todos:
        raise ValueError(
            "Replace TODO placeholders in CONFIG before running. "
            f"First placeholders found: {todos[:8]}"
        )


def load_dataset(config: PipelineConfig):
    check_placeholders(config)
    raw = pd.read_csv(config.data_path, encoding="latin1", low_memory=False)
    if config.outcome_column not in raw.columns:
        raise KeyError(f"Outcome column not found: {config.outcome_column}")

    y = map_outcome(raw[config.outcome_column], config.positive_outcome_is_success)
    frame = pd.DataFrame(index=raw.index)
    kinds: dict[str, str] = {}
    for spec in config.candidate_features:
        if spec.source_column not in raw.columns:
            raise KeyError(f"Candidate source column not found: {spec.source_column}")
        kind = spec.kind or infer_kind(raw[spec.source_column])
        if kind == "drop":
            continue
        if kind == "continuous":
            frame[spec.name] = pd.to_numeric(raw[spec.source_column], errors="coerce")
        elif kind == "binary":
            frame[spec.name] = recode_binary(raw[spec.source_column], spec.yes_no_unknown)
        elif kind == "categorical":
            frame[spec.name] = raw[spec.source_column].astype("object")
        else:
            raise ValueError(f"Unknown feature kind for {spec.name}: {kind}")
        kinds[spec.name] = kind

    keep = y.notna()
    X = frame.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)
    raw_kept = raw.loc[keep].reset_index(drop=True)
    return raw_kept, X, y, kinds


def split_kinds(kinds: dict[str, str], columns: list[str]):
    columns = list(columns)
    continuous = [c for c in columns if kinds.get(c) == "continuous"]
    binary = [c for c in columns if kinds.get(c) == "binary"]
    categorical = [c for c in columns if kinds.get(c) == "categorical"]
    return continuous, binary, categorical


def make_preprocessor(kinds: dict[str, str], columns: list[str], random_state: int):
    continuous, binary, categorical = split_kinds(kinds, columns)
    continuous_pipeline = Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=10, random_state=random_state)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipeline = Pipeline(
        [
            ("normalize_mixed_types", CategoricalStringNormalizer()),
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous_pipeline, continuous),
            ("binary", binary_pipeline, binary),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# =============================================================================
# Feature selection
# =============================================================================


def apply_training_missingness_rule(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float,
    excluded_features: set[str],
):
    missing = X_train.isna().mean()
    drop = sorted(set(missing[missing > threshold].index) | set(excluded_features))
    drop = [c for c in drop if c in X_train.columns]
    return X_train.drop(columns=drop), X_test.drop(columns=drop), drop


def construct_deduplicate(X_train: pd.DataFrame, groups: list[list[str]]) -> tuple[list[str], pd.DataFrame]:
    present = set(X_train.columns)
    rows = []
    drop: set[str] = set()
    for group_index, group in enumerate(groups, start=1):
        candidates = [c for c in group if c in present]
        if len(candidates) < 2:
            continue
        keep = sorted(
            candidates,
            key=lambda c: (
                float(X_train[c].isna().mean()),
                -float(pd.to_numeric(X_train[c], errors="coerce").var(skipna=True) or 0),
            ),
        )[0]
        for c in candidates:
            rows.append(
                {
                    "construct_id": f"construct_{group_index:02d}",
                    "feature": c,
                    "training_missing_fraction": float(X_train[c].isna().mean()),
                    "kept": c == keep,
                }
            )
            if c != keep:
                drop.add(c)
    kept = [c for c in X_train.columns if c not in drop]
    return kept, pd.DataFrame(rows)


def rank_by_permutation_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kinds: dict[str, str],
    columns: list[str],
    cv: StratifiedKFold,
    random_state: int,
    n_boot: int = 10,
    n_repeats: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    importance_rows = []
    indices = np.arange(len(X_train))
    for boot_id in range(n_boot):
        boot = rng.choice(indices, size=len(indices), replace=True)
        oob = np.setdiff1d(indices, np.unique(boot))
        if len(oob) < 100 or len(np.unique(y_train.iloc[oob])) < 2:
            _, oob = train_test_split(
                indices,
                test_size=0.25,
                stratify=y_train,
                random_state=random_state + boot_id,
            )
            boot = np.setdiff1d(indices, oob)
        rank_model = Pipeline(
            [
                ("prep", make_preprocessor(kinds, columns, random_state)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        rank_model.fit(X_train.iloc[boot][columns], y_train.iloc[boot])
        result = permutation_importance(
            rank_model,
            X_train.iloc[oob][columns],
            y_train.iloc[oob],
            n_repeats=n_repeats,
            scoring="roc_auc",
            random_state=random_state + boot_id,
            n_jobs=-1,
        )
        importance_rows.append(pd.Series(result.importances_mean, index=columns))
        print(f"Permutation-ranking bootstrap {boot_id + 1}/{n_boot}")
    importance = pd.DataFrame(importance_rows)
    ranked = importance.median().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "order": range(1, len(ranked) + 1),
            "feature": ranked.index,
            "median_permutation_importance": ranked.values,
            "median_rank": importance.rank(axis=1, ascending=False).median()[ranked.index].values,
        }
    )


def encoded_feature_to_original(encoded_name: str, original_columns: list[str]) -> str:
    core = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    matches = [c for c in original_columns if core == c or core.startswith(c + "_")]
    return max(matches, key=len) if matches else core


def lasso_boruta_union(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kinds: dict[str, str],
    ranked_columns: list[str],
    cv: StratifiedKFold,
    random_state: int,
) -> dict[str, list[str]]:
    prep = make_preprocessor(kinds, ranked_columns, random_state)
    encoded = prep.fit_transform(X_train[ranked_columns], y_train)
    encoded = np.asarray(encoded)
    feature_names = list(prep.get_feature_names_out())
    original_by_encoded = [
        encoded_feature_to_original(name, ranked_columns) for name in feature_names
    ]

    lasso = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        max_iter=3000,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    lasso.fit(encoded, y_train)
    lasso_vars = sorted(
        {original_by_encoded[i] for i, coef in enumerate(lasso.coef_.ravel()) if abs(coef) > 1e-8}
    )

    boruta_vars: list[str] = []
    if BorutaPy is not None:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        boruta = BorutaPy(
            rf,
            n_estimators="auto",
            perc=90,
            max_iter=40,
            random_state=random_state,
            verbose=0,
        )
        boruta.fit(encoded, np.asarray(y_train))
        mask = boruta.support_ | boruta.support_weak_
        boruta_vars = sorted({original_by_encoded[i] for i, keep in enumerate(mask) if keep})
    else:
        print("Boruta is not installed; using LASSO-only feature union.")

    union = sorted(set(lasso_vars) | set(boruta_vars), key=lambda c: ranked_columns.index(c))
    return {"lasso": lasso_vars, "boruta": boruta_vars, "union": union}


def select_top_construct_unique(
    primary_features: list[str],
    ranked_pool: list[str],
    construct_groups: list[list[str]],
    n_features: int,
) -> list[str]:
    construct_lookup: dict[str, str] = {}
    for i, group in enumerate(construct_groups, start=1):
        for feature in group:
            construct_lookup[feature] = f"construct_{i:02d}"

    selected: list[str] = []
    used_constructs: set[str] = set()
    for feature in list(primary_features) + [c for c in ranked_pool if c not in primary_features]:
        construct_id = construct_lookup.get(feature, f"singleton::{feature}")
        if construct_id in used_constructs:
            continue
        selected.append(feature)
        used_constructs.add(construct_id)
        if len(selected) >= n_features:
            break
    return selected


# =============================================================================
# Model comparison and tuning
# =============================================================================


def model_families(random_state: int, negative_positive_ratio: float):
    return {
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        ),
        "RidgeLogistic": LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=3000, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(),
        "BaggingTrees": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            random_state=random_state,
            n_jobs=1,
        ),
        "NeuralNetwork": MLPClassifier(
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=random_state,
        ),
        "SVM": SVC(probability=True, random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(random_state=random_state, n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            random_state=random_state,
        ),
        "GaussianNB": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }


def optuna_params(model_name: str, trial, scale_pos_weight: float):
    if model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 1500, step=100),
            "model__max_depth": trial.suggest_int("model__max_depth", 2, 7),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.005, 0.05, log=True),
            "model__min_child_weight": trial.suggest_float("model__min_child_weight", 5.0, 60.0, log=True),
            "model__gamma": trial.suggest_float("model__gamma", 0.0, 8.0),
            "model__subsample": trial.suggest_float("model__subsample", 0.75, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.75, 1.0),
            "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 1.0, log=True),
            "model__reg_lambda": trial.suggest_float("model__reg_lambda", 3.0, 60.0, log=True),
            "model__scale_pos_weight": trial.suggest_float(
                "model__scale_pos_weight", 0.8, max(6.0, scale_pos_weight * 1.5), log=True
            ),
        }
    if model_name == "RidgeLogistic":
        return {
            "model__C": trial.suggest_float("model__C", 0.01, 10.0, log=True),
            "model__class_weight": trial.suggest_categorical("model__class_weight", [None, "balanced"]),
        }
    if model_name == "KNN":
        return {
            "model__n_neighbors": trial.suggest_int("model__n_neighbors", 7, 75),
            "model__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"]),
            "model__p": trial.suggest_categorical("model__p", [1, 2]),
        }
    if model_name == "BaggingTrees":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 400, step=50),
            "model__max_samples": trial.suggest_float("model__max_samples", 0.6, 1.0),
            "model__bootstrap": trial.suggest_categorical("model__bootstrap", [True, False]),
            "model__estimator__max_depth": trial.suggest_categorical(
                "model__estimator__max_depth", [2, 3, 4, 5, 6, None]
            ),
            "model__estimator__min_samples_leaf": trial.suggest_int(
                "model__estimator__min_samples_leaf", 10, 80
            ),
            "model__estimator__class_weight": trial.suggest_categorical(
                "model__estimator__class_weight", [None, "balanced"]
            ),
        }
    if model_name == "NeuralNetwork":
        return {
            "model__hidden_layer_sizes": trial.suggest_categorical(
                "model__hidden_layer_sizes", [(8,), (16,), (32,), (16, 8), (32, 16)]
            ),
            "model__activation": trial.suggest_categorical("model__activation", ["relu", "tanh"]),
            "model__alpha": trial.suggest_float("model__alpha", 1e-5, 1e-2, log=True),
            "model__learning_rate_init": trial.suggest_float(
                "model__learning_rate_init", 1e-4, 3e-3, log=True
            ),
        }
    if model_name == "SVM":
        return {
            "model__kernel": trial.suggest_categorical("model__kernel", ["linear", "rbf", "poly"]),
            "model__C": trial.suggest_float("model__C", 0.01, 10.0, log=True),
            "model__gamma": trial.suggest_categorical("model__gamma", ["scale", "auto"]),
            "model__degree": trial.suggest_int("model__degree", 2, 3),
            "model__class_weight": trial.suggest_categorical("model__class_weight", [None, "balanced"]),
        }
    if model_name == "ExtraTrees":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 150, 800, step=50),
            "model__max_depth": trial.suggest_categorical("model__max_depth", [2, 3, 4, 5, 7, None]),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 5, 80),
            "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", None]),
            "model__class_weight": trial.suggest_categorical("model__class_weight", [None, "balanced"]),
        }
    if model_name == "AdaBoost":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 50, 400, step=50),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.005, 0.3, log=True),
            "model__estimator__max_depth": trial.suggest_int("model__estimator__max_depth", 1, 3),
            "model__estimator__min_samples_leaf": trial.suggest_int(
                "model__estimator__min_samples_leaf", 5, 50
            ),
        }
    if model_name == "GaussianNB":
        return {"model__var_smoothing": trial.suggest_float("model__var_smoothing", 1e-12, 1e-6, log=True)}
    raise ValueError(f"Unsupported model for Optuna: {model_name}")


def tune_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kinds: dict[str, str],
    features: list[str],
    cv: StratifiedKFold,
    config: PipelineConfig,
    scale_pos_weight: float,
):
    pipeline = Pipeline(
        [
            ("prep", make_preprocessor(kinds, features, config.random_state)),
            ("model", estimator),
        ]
    )
    if config.use_optuna and optuna is not None and model_name != "QDA":
        def objective(trial):
            params = optuna_params(model_name, trial, scale_pos_weight)
            candidate = clone(pipeline).set_params(**params)
            scores = cross_validate(
                candidate,
                X_train[features],
                y_train,
                cv=cv,
                scoring={"auc": "roc_auc", "ap": "average_precision"},
                n_jobs=1,
                error_score=np.nan,
            )
            auc = float(np.nanmean(scores["test_auc"]))
            trial.set_user_attr("mean_auc", auc)
            trial.set_user_attr("mean_ap", float(np.nanmean(scores["test_ap"])))
            trial.set_user_attr("params", params)
            return -1.0 if np.isnan(auc) else auc

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_state),
        )
        study.optimize(objective, n_trials=config.optuna_trials_5var, show_progress_bar=False)
        best_params = study.best_trial.user_attrs["params"]
        best_pipeline = clone(pipeline).set_params(**best_params)
        cv_auc = float(study.best_trial.user_attrs["mean_auc"])
        cv_ap = float(study.best_trial.user_attrs["mean_ap"])
    else:
        scores = cross_validate(
            pipeline,
            X_train[features],
            y_train,
            cv=cv,
            scoring={"auc": "roc_auc", "ap": "average_precision"},
            n_jobs=1,
            error_score=np.nan,
        )
        best_pipeline = pipeline
        best_params = {}
        cv_auc = float(np.nanmean(scores["test_auc"]))
        cv_ap = float(np.nanmean(scores["test_ap"]))
    best_pipeline.fit(X_train[features], y_train)
    return best_pipeline, best_params, cv_auc, cv_ap


def model_specific_swap_search(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kinds: dict[str, str],
    selected: list[str],
    ranked_pool: list[str],
    cv: StratifiedKFold,
    config: PipelineConfig,
) -> list[str]:
    current = list(selected)
    pool = [c for c in ranked_pool if c in X_train.columns]

    def cv_auc(features):
        pipe = Pipeline(
            [
                ("prep", make_preprocessor(kinds, features, config.random_state)),
                ("model", clone(estimator)),
            ]
        )
        scores = cross_validate(
            pipe,
            X_train[features],
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            error_score=np.nan,
        )
        return float(np.nanmean(scores["test_score"]))

    best_score = cv_auc(current)
    attempts = 0
    improved = True
    while improved and attempts < config.model_swap_cycles:
        improved = False
        for old in list(current):
            for new in pool:
                if new in current:
                    continue
                candidate = [new if f == old else f for f in current]
                attempts += 1
                score = cv_auc(candidate)
                if score > best_score:
                    current, best_score, improved = candidate, score, True
                    print(f"Swap improved CV AUC to {best_score:.4f}: {old} -> {new}")
                if attempts >= config.model_swap_cycles:
                    break
            if attempts >= config.model_swap_cycles:
                break
    return current


# =============================================================================
# Metrics, clinical scores, fairness, and LOCO
# =============================================================================


def bootstrap_metric_ci(y, p, metric_fn: Callable, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan
    values = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        values.append(metric_fn(y[idx], p[idx]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def metric_row(name: str, y, p, config: PipelineConfig):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, p) if len(y) else np.nan
    auc_low, auc_high = (
        bootstrap_metric_ci(y, p, roc_auc_score, config.n_boot, config.random_state + 100)
        if np.isfinite(auc)
        else (np.nan, np.nan)
    )
    return {
        "model": name,
        "n": int(len(y)),
        "success_rate": float(np.mean(y)) if len(y) else np.nan,
        "auc": auc,
        "auc_low": auc_low,
        "auc_high": auc_high,
        "average_precision": ap,
        "brier_score": brier,
    }


def optional_numeric(raw: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.to_numeric(raw[column], errors="coerce")


def complete_case_sum(parts: list[pd.Series]) -> pd.Series:
    return pd.concat(parts, axis=1).sum(axis=1, skipna=False)


def compute_jcto(raw: pd.DataFrame, score_cols: ScoreColumns) -> pd.Series:
    return complete_case_sum(
        [
            optional_numeric(raw, score_cols.jcto_stump),
            optional_numeric(raw, score_cols.jcto_calcification),
            optional_numeric(raw, score_cols.jcto_tortuosity),
            optional_numeric(raw, score_cols.jcto_length),
            optional_numeric(raw, score_cols.jcto_prior_attempt),
        ]
    )


def compute_progress_cto(raw: pd.DataFrame, score_cols: ScoreColumns) -> pd.Series:
    return complete_case_sum(
        [
            optional_numeric(raw, score_cols.progress_proximal_cap),
            optional_numeric(raw, score_cols.progress_collateral),
            optional_numeric(raw, score_cols.progress_tortuosity),
            optional_numeric(raw, score_cols.progress_circumflex),
        ]
    )


def compute_castle(raw: pd.DataFrame, score_cols: ScoreColumns) -> pd.Series:
    age = optional_numeric(raw, score_cols.castle_age_years)
    length_mm = optional_numeric(raw, score_cols.castle_length_mm)
    return complete_case_sum(
        [
            (optional_numeric(raw, score_cols.castle_prior_bypass) == 1).astype(float),
            (age >= 70).astype(float).where(age.notna(), np.nan),
            (optional_numeric(raw, score_cols.castle_stump_adverse) == 1).astype(float),
            (optional_numeric(raw, score_cols.castle_tortuosity) == 1).astype(float),
            (length_mm >= 20).astype(float).where(length_mm.notna(), np.nan),
            (optional_numeric(raw, score_cols.castle_calcification) == 1).astype(float),
        ]
    )


def evaluate_clinical_scores(raw_test: pd.DataFrame, y_test: pd.Series, config: PipelineConfig):
    rows = []
    for name, score in {
        "J-CTO": compute_jcto(raw_test, config.score_columns),
        "PROGRESS-CTO": compute_progress_cto(raw_test, config.score_columns),
        "CASTLE": compute_castle(raw_test, config.score_columns),
    }.items():
        rows.append(metric_row(name, y_test, -score.to_numpy(dtype=float), config))
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    y = np.asarray(y).astype(int)
    lp = logit(p)

    def nll(theta):
        intercept, slope = theta
        q = 1 / (1 + np.exp(-(intercept + slope * lp)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    result = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return tuple(result.x) if result.success else (np.nan, np.nan)


def label_sex(value):
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().lower()
    if text in {"0", "female", "f", "woman"}:
        return "Female"
    if text in {"1", "male", "m", "man"}:
        return "Male"
    return str(value)


def fairness_metrics(raw_test: pd.DataFrame, y_test: pd.Series, p, config: PipelineConfig):
    rows = []
    axes: dict[str, pd.Series] = {}
    if config.score_columns.sex and config.score_columns.sex in raw_test.columns:
        axes["Sex"] = raw_test[config.score_columns.sex].map(label_sex)
    if config.score_columns.age_years and config.score_columns.age_years in raw_test.columns:
        age = pd.to_numeric(raw_test[config.score_columns.age_years], errors="coerce")
        axes["Age group"] = pd.Series(np.where(age >= 70, ">=70", "<70"), index=raw_test.index)
        axes["Age group"] = axes["Age group"].where(age.notna(), "Missing")

    frame = pd.DataFrame({"y": np.asarray(y_test), "p": np.asarray(p)}, index=raw_test.index)
    for axis_name, labels in axes.items():
        frame[axis_name] = labels
        for subgroup, group in frame.groupby(axis_name, dropna=False):
            if len(group) < 10 or group["y"].nunique() < 2:
                continue
            row = metric_row("Representative model", group["y"], group["p"], config)
            intercept, slope = calibration_intercept_slope(group["y"], group["p"])
            row.update(
                {
                    "fairness_axis": axis_name,
                    "subgroup": subgroup,
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_loco(
    X: pd.DataFrame,
    y: pd.Series,
    raw: pd.DataFrame,
    kinds: dict[str, str],
    features: list[str],
    best_estimator,
    config: PipelineConfig,
):
    center_column = config.score_columns.center_id
    if center_column is None or center_column not in raw.columns:
        raise KeyError("LOCO requires score_columns.center_id.")
    data = pd.concat([X[features], y.rename("y"), raw[center_column].rename("center_id")], axis=1)
    data = data.dropna(subset=["center_id"]).copy()
    rows, predictions = [], []
    for i, center in enumerate(sorted(data["center_id"].dropna().unique()), start=1):
        train = data.loc[data["center_id"] != center]
        test = data.loc[data["center_id"] == center]
        y_train = train["y"].astype(int)
        y_test = test["y"].astype(int)
        row = {
            "center_id": center,
            "n_train": int(len(train)),
            "n_test_center": int(len(test)),
            "n_success_center": int(y_test.sum()),
            "n_failure_center": int(len(y_test) - y_test.sum()),
        }
        if len(test) < 10 or y_test.nunique() < 2 or row["n_success_center"] < 2 or row["n_failure_center"] < 2:
            row.update({"auc": np.nan, "average_precision": np.nan, "brier_score": np.nan, "note": "Not estimable"})
            rows.append(row)
            continue
        model = Pipeline(
            [
                ("prep", make_preprocessor(kinds, features, config.random_state)),
                ("model", clone(best_estimator)),
            ]
        )
        model.fit(train[features], y_train)
        p = model.predict_proba(test[features])[:, 1]
        row.update(metric_row("LOCO representative model", y_test, p, config))
        row["note"] = ""
        rows.append(row)
        predictions.append(pd.DataFrame({"center_id": center, "row_index": test.index, "y_true": y_test, "p_loco": p}))
        if i % 10 == 0:
            print(f"Completed {i} LOCO centers")
    by_center = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "n_centers_total": int(len(by_center)),
                "n_centers_auc_estimable": int(by_center["auc"].notna().sum()) if "auc" in by_center else 0,
                "n_loco_predictions": int(len(pred)),
                "pooled_auc": roc_auc_score(pred["y_true"], pred["p_loco"])
                if len(pred) and pred["y_true"].nunique() == 2
                else np.nan,
                "pooled_average_precision": average_precision_score(pred["y_true"], pred["p_loco"])
                if len(pred) and pred["y_true"].nunique() == 2
                else np.nan,
                "pooled_brier_score": brier_score_loss(pred["y_true"], pred["p_loco"]) if len(pred) else np.nan,
                "median_center_auc": float(by_center["auc"].median()) if "auc" in by_center else np.nan,
            }
        ]
    )
    return by_center, summary, pred


# =============================================================================
# Driver
# =============================================================================


def run_pipeline(config: PipelineConfig = CONFIG):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw, X, y, kinds = load_dataset(config)
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X, y, raw, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    X_train, X_test, dropped_missing = apply_training_missingness_rule(
        X_train,
        X_test,
        config.missingness_threshold,
        config.leakage_or_excluded_features,
    )
    kinds = {k: v for k, v in kinds.items() if k in X_train.columns}

    dedup_kept, dedup_table = construct_deduplicate(X_train, config.construct_groups)
    dedup_table.to_csv(config.output_dir / "construct_deduplication.csv", index=False)

    cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
    ranking = rank_by_permutation_importance(
        X_train,
        y_train,
        kinds,
        dedup_kept,
        cv,
        config.random_state,
    )
    ranking.to_csv(config.output_dir / "feature_ranking.csv", index=False)
    ranked_pool = ranking["feature"].tolist()

    selection = lasso_boruta_union(X_train, y_train, kinds, ranked_pool, cv, config.random_state)
    pd.DataFrame(
        [{"stage": stage, "feature": feature} for stage, values in selection.items() for feature in values]
    ).to_csv(config.output_dir / "feature_selection_lasso_boruta.csv", index=False)

    selected_features = select_top_construct_unique(
        selection["union"], ranked_pool, config.construct_groups, config.final_feature_count
    )
    pd.Series(selected_features, name="feature").to_csv(
        config.output_dir / "selected_features_initial.csv", index=False
    )

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / max(1, positive)
    families = {k: v for k, v in model_families(config.random_state, scale_pos_weight).items() if v is not None}

    fitted_models = {}
    results = []
    model_specific_features = {}
    for model_name, estimator in families.items():
        features = selected_features
        if config.run_model_specific_swap_search:
            features = model_specific_swap_search(
                estimator, X_train, y_train, kinds, selected_features, ranked_pool, cv, config
            )
        model_specific_features[model_name] = features
        model, params, cv_auc, cv_ap = tune_model(
            model_name,
            estimator,
            X_train,
            y_train,
            kinds,
            features,
            cv,
            config,
            scale_pos_weight,
        )
        p_test = model.predict_proba(X_test[features])[:, 1]
        row = metric_row(model_name, y_test, p_test, config)
        row.update({"cv_auc": cv_auc, "cv_average_precision": cv_ap})
        results.append(row)
        fitted_models[model_name] = {"model": model, "features": features, "params": params, "p_test": p_test}
        print(f"{model_name}: CV AUC {cv_auc:.4f}; test AUC {row['auc']:.4f}")

    comparison = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
    comparison.to_csv(config.output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        [{"model": name, "feature": feature} for name, values in model_specific_features.items() for feature in values]
    ).to_csv(config.output_dir / "model_specific_selected_features.csv", index=False)

    representative_name = str(comparison.iloc[0]["model"])
    representative = fitted_models[representative_name]
    representative_features = representative["features"]
    base_model = representative["model"]

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=config.cv_splits)
    calibrated.fit(X_train[representative_features], y_train)
    p_calibrated = calibrated.predict_proba(X_test[representative_features])[:, 1]
    calibrated_metrics = pd.DataFrame([metric_row(f"{representative_name} isotonic calibrated", y_test, p_calibrated, config)])
    calibrated_metrics.to_csv(config.output_dir / "calibrated_model_performance.csv", index=False)
    joblib.dump(calibrated, config.output_dir / "Technical_success_model.joblib")

    predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "observed_outcome": y_test.to_numpy(),
            "representative_probability": representative["p_test"],
            "calibrated_probability": p_calibrated,
        }
    )
    predictions.to_csv(config.output_dir / "heldout_predictions.csv", index=False)

    score_table = evaluate_clinical_scores(raw_test, y_test, config)
    score_table.to_csv(config.output_dir / "clinical_score_performance.csv", index=False)

    fairness = fairness_metrics(raw_test, y_test, p_calibrated, config)
    fairness.to_csv(config.output_dir / "fairness_subgroup_metrics.csv", index=False)

    if config.run_loco:
        loco_by_center, loco_summary, loco_predictions = run_loco(
            X, y, raw, kinds, representative_features, base_model.named_steps["model"], config
        )
        loco_by_center.to_csv(config.output_dir / "loco_by_center.csv", index=False)
        loco_summary.to_csv(config.output_dir / "loco_summary.csv", index=False)
        loco_predictions.to_csv(config.output_dir / "loco_predictions.csv", index=False)

    metadata = {
        "positive_class": "technical success" if config.positive_outcome_is_success else "technical failure",
        "candidate_feature_count_after_cleaning": int(X_train.shape[1]),
        "dropped_by_missingness_or_exclusion": dropped_missing,
        "initial_selected_features": selected_features,
        "model_specific_selected_features": model_specific_features,
        "representative_model": representative_name,
        "representative_features": representative_features,
        "test_size": config.test_size,
        "cv_splits": config.cv_splits,
        "clinical_score_comparators": ["J-CTO", "PROGRESS-CTO", "CASTLE"],
    }
    (config.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "comparison": comparison,
        "calibrated_metrics": calibrated_metrics,
        "clinical_scores": score_table,
        "fairness": fairness,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_pipeline(CONFIG)

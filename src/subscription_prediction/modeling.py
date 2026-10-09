"""Preprocessing is fitted on training data within each pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X, estimator):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def candidate_models(config):
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=config["random_state"]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config["random_forest_estimators"],
            max_depth=config["random_forest_max_depth"],
            min_samples_leaf=3, class_weight="balanced",
            random_state=config["random_state"], n_jobs=-1,
        ),
    }

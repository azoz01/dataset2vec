import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessing_pipeline() -> BaseEstimator:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one-hot",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    ).set_output(transform="pandas")

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    ).set_output(transform="pandas")

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(
                            dtype_include=("object", "category")
                        ),
                    ),
                    (
                        num_pipeline,
                        make_column_selector(dtype_include=np.number),
                    ),
                ),
            )
        ]
    ).set_output(transform="pandas")
    return pipeline


def is_positive(number: int) -> int:
    assert number > 0, "Number is non-positive"
    return number


def all_elements_positive(arr: list[int]) -> list[int]:
    assert all(
        map(lambda x: x > 0, arr)
    ), "List contains non-positive elements"
    return arr


def non_empty(arr: list[int]) -> list[int]:
    assert len(arr) > 0, "List is empty"
    return arr

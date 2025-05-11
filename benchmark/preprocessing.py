"""Preprocessing pipeline builder for scaling, encoding, and imputation."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder

from .utils import setup_logger


class PreprocessingPipeline:
    """Builds and manages sklearn preprocessing pipelines from config."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger("PreprocessingPipeline")
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_pipeline: Optional[Pipeline] = None
        self._fitted = False

    def build(
        self,
        X: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
    ) -> "PreprocessingPipeline":
        """Build a preprocessing pipeline tailored to the input data."""
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if numeric_columns is None:
            numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

        transformers = []

        # Numeric pipeline: imputation + scaling
        if numeric_columns:
            numeric_steps = []
            if self.config.get("impute", True):
                strategy = self.config.get("impute_strategy", "median")
                numeric_steps.append(("imputer", SimpleImputer(strategy=strategy)))

            scaler_name = self.config.get("scale", "standard")
            if scaler_name is True:
                scaler_name = "standard"
            elif scaler_name is False:
                scaler_name = "none"

            if scaler_name == "standard":
                numeric_steps.append(("scaler", StandardScaler()))
            elif scaler_name == "minmax":
                numeric_steps.append(("scaler", MinMaxScaler()))
            elif scaler_name not in (None, "none"):
                raise ValueError(f"Unsupported scaler: {scaler_name}")

            if numeric_steps:
                transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
                self.logger.info(f"Numeric pipeline for {len(numeric_columns)} columns")

        # Categorical pipeline: imputation + one-hot encoding
        if categorical_columns:
            cat_steps = []
            if self.config.get("impute", True):
                cat_steps.append(
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                )

            if self.config.get("encode", True):
                cat_steps.append(
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    )
                )

            if cat_steps:
                transformers.append(("categorical", Pipeline(cat_steps), categorical_columns))
                self.logger.info(f"Categorical pipeline for {len(categorical_columns)} columns")

        if transformers:
            self.feature_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=False,
            )
            self.feature_pipeline.set_output(transform="pandas")
        else:
            self.feature_pipeline = None

        return self

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and transform features."""
        if self.feature_pipeline is None:
            self.logger.info("No preprocessing needed; returning data as-is")
            return X.copy()
        X_transformed = self.feature_pipeline.fit_transform(X)
        self._fitted = True
        self.logger.info(f"Transformed features to shape {X_transformed.shape}")
        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using a fitted pipeline."""
        if self.feature_pipeline is None:
            return X.copy()
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit_transform first.")
        return self.feature_pipeline.transform(X)

    def fit_transform_target(self, y: pd.Series) -> pd.Series:
        """Optionally encode the target variable."""
        if self.config.get("encode_target", False):
            self.target_encoder = LabelEncoder()
            y_transformed = pd.Series(
                self.target_encoder.fit_transform(y),
                index=y.index,
                name=y.name,
            )
            self.logger.info("Target label-encoded")
            return y_transformed
        return y.copy()

    def transform_target(self, y: pd.Series) -> pd.Series:
        """Encode target using a fitted encoder."""
        if self.target_encoder is None:
            return y.copy()
        return pd.Series(
            self.target_encoder.transform(y),
            index=y.index,
            name=y.name,
        )

    def inverse_transform_target(self, y: pd.Series) -> pd.Series:
        """Decode target labels back to original values."""
        if self.target_encoder is None:
            return y.copy()
        return pd.Series(
            self.target_encoder.inverse_transform(y),
            index=y.index,
            name=y.name,
        )


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[PreprocessingPipeline, pd.DataFrame]:
    """Convenience factory: build pipeline, fit_transform features, and return both."""
    pipeline = PreprocessingPipeline(config).build(X)
    X_transformed = pipeline.fit_transform(X)
    return pipeline, X_transformed

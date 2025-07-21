import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Optional, Tuple

from .data_validator import DataValidator, ValidationLevel, ValidationResult


class DataPreprocessor:
    def __init__(self, scaler=None, enable_validation: bool = True, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """Preprocess raw sensor data.

        Parameters
        ----------
        scaler : sklearn-like scaler or None, optional
            Scaler used for normalization. Defaults to ``MinMaxScaler``.
        enable_validation : bool, default True
            Whether to enable comprehensive data validation before preprocessing.
        validation_level : ValidationLevel, default ValidationLevel.MODERATE
            Validation strictness level.
        """
        self.scaler = scaler or MinMaxScaler()
        self.enable_validation = enable_validation
        self.validator = DataValidator(validation_level) if enable_validation else None

    def save(self, path: str) -> None:
        """Persist the underlying scaler to ``path``."""
        try:
            joblib.dump(self.scaler, Path(path))
            logging.info(f"Scaler saved successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to save scaler to {path}: {e}")
            raise ValueError(f"Unable to save scaler to {path}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load a scaler from ``path`` and return a new ``DataPreprocessor``."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        
        try:
            scaler = joblib.load(path_obj)
            logging.info(f"Scaler loaded successfully from {path}")
            return cls(scaler)
        except Exception as e:
            logging.error(f"Failed to load scaler from {path}: {e}")
            raise ValueError(f"Unable to load scaler from {path}: {e}") from e

    def validate_data(self, df: pd.DataFrame, auto_fix: bool = False) -> Tuple[ValidationResult, pd.DataFrame]:
        """
        Validate DataFrame using comprehensive data validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        auto_fix : bool, default False
            Whether to automatically fix detected issues
            
        Returns
        -------
        Tuple[ValidationResult, pd.DataFrame]
            Validation results and potentially fixed DataFrame
        """
        if not self.enable_validation:
            logging.info("Data validation is disabled, skipping validation")
            return ValidationResult(True, [], [], [], {}), df
        
        logging.info("Running comprehensive data validation...")
        
        # Schema validation
        schema_result = self.validator.validate_schema(df)
        
        # Data quality validation  
        quality_result = self.validator.validate_data_quality(df)
        
        # Time series validation (without time column for sensor data)
        ts_result = self.validator.validate_time_series_properties(df)
        
        # Auto-fix if requested
        fixed_df = df
        auto_fixes = []
        if auto_fix and self.validator.validation_level != ValidationLevel.STRICT:
            fixed_df, auto_fixes = self.validator.auto_fix_data(df)
            if auto_fixes:
                logging.info(f"Auto-fixes applied: {auto_fixes}")
        
        # Combine results
        all_errors = schema_result.errors + quality_result.errors + ts_result.errors
        all_warnings = schema_result.warnings + quality_result.warnings + ts_result.warnings
        all_fixes = schema_result.fixed_issues + quality_result.fixed_issues + ts_result.fixed_issues + auto_fixes
        
        combined_summary = {
            "schema_validation": schema_result.summary,
            "quality_validation": quality_result.summary,
            "time_series_validation": ts_result.summary,
            "validation_level": self.validator.validation_level.value,
            "auto_fixes_applied": auto_fixes
        }
        
        is_valid = len(all_errors) == 0
        result = ValidationResult(is_valid, all_errors, all_warnings, all_fixes, combined_summary)
        
        if not is_valid:
            logging.warning(f"Data validation found {len(all_errors)} errors and {len(all_warnings)} warnings")
        else:
            logging.info("Data validation passed successfully")
        
        return result, fixed_df

    def fit_transform(self, df: pd.DataFrame, validate: bool = True, auto_fix: bool = False) -> np.ndarray:
        """
        Fit scaler to data and transform it.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fit and transform
        validate : bool, default True
            Whether to validate data before processing
        auto_fix : bool, default False
            Whether to automatically fix detected issues
            
        Returns
        -------
        np.ndarray
            Scaled data array
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Run validation if enabled
        if validate and self.enable_validation:
            validation_result, df = self.validate_data(df, auto_fix=auto_fix)
            
            # Raise error if validation fails and auto_fix didn't resolve issues
            if not validation_result.is_valid:
                error_msg = f"Data validation failed with {len(validation_result.errors)} errors: " + \
                           "; ".join(validation_result.errors[:3])  # Show first 3 errors
                if len(validation_result.errors) > 3:
                    error_msg += f" (and {len(validation_result.errors) - 3} more)"
                raise ValueError(error_msg)
        
        # Legacy validation for backward compatibility
        if df.isnull().any().any():
            raise ValueError("DataFrame contains missing values")
        
        try:
            scaled = self.scaler.fit_transform(df)
            logging.info(f"Data fitted and transformed: shape {scaled.shape}")
            return scaled
        except Exception as e:
            logging.error(f"Failed to fit and transform data: {e}")
            raise ValueError(f"Unable to fit and transform data: {e}") from e

    def transform(self, df: pd.DataFrame, validate: bool = True, auto_fix: bool = False) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform
        validate : bool, default True
            Whether to validate data before processing
        auto_fix : bool, default False
            Whether to automatically fix detected issues
            
        Returns
        -------
        np.ndarray
            Scaled data array
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Run validation if enabled
        if validate and self.enable_validation:
            validation_result, df = self.validate_data(df, auto_fix=auto_fix)
            
            # Raise error if validation fails and auto_fix didn't resolve issues
            if not validation_result.is_valid:
                error_msg = f"Data validation failed with {len(validation_result.errors)} errors: " + \
                           "; ".join(validation_result.errors[:3])  # Show first 3 errors
                if len(validation_result.errors) > 3:
                    error_msg += f" (and {len(validation_result.errors) - 3} more)"
                raise ValueError(error_msg)
        
        # Legacy validation for backward compatibility
        if df.isnull().any().any():
            raise ValueError("DataFrame contains missing values")
        
        try:
            transformed = self.scaler.transform(df)
            logging.info(f"Data transformed: shape {transformed.shape}")
            return transformed
        except Exception as e:
            logging.error(f"Failed to transform data: {e}")
            raise ValueError(f"Unable to transform data: {e}") from e

    def create_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        """Create sliding windows from data."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if step <= 0:
            raise ValueError("step must be positive")
        
        if window_size > len(data):
            raise ValueError("window_size cannot be larger than data length")
        
        try:
            windows = []
            for start in range(0, len(data) - window_size + 1, step):
                windows.append(data[start:start + window_size])
            
            if not windows:
                raise ValueError("No windows could be created with given parameters")
            
            result = np.stack(windows)
            logging.info(f"Created {len(windows)} windows with shape {result.shape}")
            return result
        except Exception as e:
            logging.error(f"Failed to create windows: {e}")
            raise ValueError(f"Unable to create windows: {e}") from e

    def load_and_preprocess(self, csv_path: str, window_size: int, step: int = 1) -> np.ndarray:
        """Load CSV data, preprocess it, and create windows."""
        csv_path_obj = Path(csv_path)
        
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path_obj)
            logging.info(f"Loaded CSV file: {csv_path}, shape: {df.shape}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Unable to parse CSV file {csv_path}: {e}")
        except Exception as e:
            raise ValueError(f"Unable to read CSV file {csv_path}: {e}")
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        scaled = self.fit_transform(df)
        return self.create_windows(scaled, window_size, step)

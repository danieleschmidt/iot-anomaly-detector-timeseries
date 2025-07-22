import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
import logging
import time
from typing import Optional, Tuple, Generator, Iterator, Dict, Any

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
    
    def create_sliding_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        """Create sliding windows from data with preprocessing.
        
        This method applies scaling and creates sliding windows in one step,
        optimized for streaming data processing.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        window_size : int
            Size of each window
        step : int, default 1
            Step size between windows
            
        Returns
        -------
        np.ndarray
            Preprocessed sliding windows
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if step <= 0:
            raise ValueError("step must be positive")
        
        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
        
        try:
            # Apply scaling if not already fitted
            if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
                logging.info("Fitting scaler on streaming data")
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)
            
            # Create sliding windows
            windows = []
            for start in range(0, len(scaled_data) - window_size + 1, step):
                windows.append(scaled_data[start:start + window_size])
            
            if not windows:
                raise ValueError("No windows could be created with given parameters")
            
            result = np.stack(windows)
            logging.debug(f"Created {len(windows)} sliding windows with shape {result.shape}")
            return result
            
        except Exception as e:
            logging.error(f"Failed to create sliding windows: {e}")
            raise ValueError(f"Unable to create sliding windows: {e}") from e
    
    def create_windows_generator(
        self, 
        data: np.ndarray, 
        window_size: int, 
        step: int = 1
    ) -> Generator[np.ndarray, None, None]:
        """Create sliding windows from data using a memory-efficient generator.
        
        This generator yields windows on-demand rather than creating all windows
        at once, significantly reducing memory usage for large datasets.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        window_size : int
            Size of each window
        step : int, default 1
            Step size between windows
            
        Yields
        ------
        np.ndarray
            Individual windows of shape (window_size, n_features)
            
        Raises
        ------
        ValueError
            If parameters are invalid or insufficient data
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        if step <= 0:
            raise ValueError("Step size must be positive")
        
        if window_size > len(data):
            raise ValueError("Window size cannot be larger than data length")
        
        logging.debug(f"Creating window generator: window_size={window_size}, step={step}")
        
        # Generate windows on-demand
        for start in range(0, len(data) - window_size + 1, step):
            yield data[start:start + window_size].copy()
    
    def create_sliding_windows_generator(
        self,
        data: np.ndarray,
        window_size: int,
        step: int = 1
    ) -> Generator[np.ndarray, None, None]:
        """Create sliding windows with preprocessing using a memory-efficient generator.
        
        This method applies scaling and creates sliding windows on-demand,
        optimized for very large datasets that don't fit in memory.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        window_size : int
            Size of each window
        step : int, default 1
            Step size between windows
            
        Yields
        ------
        np.ndarray
            Preprocessed windows of shape (window_size, n_features)
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        if step <= 0:
            raise ValueError("Step size must be positive")
        
        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
        
        try:
            # Apply scaling if not already fitted
            if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
                logging.info("Fitting scaler on data for generator")
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)
            
            logging.debug(f"Creating sliding window generator: window_size={window_size}, step={step}")
            
            # Generate preprocessed windows on-demand
            for start in range(0, len(scaled_data) - window_size + 1, step):
                yield scaled_data[start:start + window_size].copy()
                
        except Exception as e:
            logging.error(f"Failed to create sliding windows generator: {e}")
            raise ValueError(f"Unable to create sliding windows generator: {e}") from e
    
    def process_windows_batched(
        self,
        window_generator: Generator[np.ndarray, None, None],
        batch_size: int
    ) -> Generator[np.ndarray, None, None]:
        """Process windows from generator in batches for memory efficiency.
        
        Parameters
        ----------
        window_generator : Generator
            Generator yielding individual windows
        batch_size : int
            Number of windows to process in each batch
            
        Yields
        ------
        np.ndarray
            Batches of windows with shape (batch_size, window_size, n_features)
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        batch = []
        
        try:
            for window in window_generator:
                batch.append(window)
                
                if len(batch) == batch_size:
                    yield np.stack(batch)
                    batch = []
            
            # Yield final partial batch if any windows remain
            if batch:
                yield np.stack(batch)
                
        except Exception as e:
            logging.error(f"Error in batched window processing: {e}")
            raise ValueError(f"Failed to process windows in batches: {e}") from e
    
    def estimate_window_memory_usage(
        self,
        data_shape: Tuple[int, int],
        window_size: int,
        batch_size: int = 256,
        dtype: np.dtype = np.float32
    ) -> Dict[str, float]:
        """Estimate memory usage for different window creation approaches.
        
        Parameters
        ----------
        data_shape : Tuple[int, int]
            Shape of input data (n_samples, n_features)
        window_size : int
            Size of each window
        batch_size : int, default 256
            Batch size for generator approach
        dtype : np.dtype, default np.float32
            Data type for memory calculations
            
        Returns
        -------
        Dict[str, float]
            Memory usage estimates in MB
        """
        n_samples, n_features = data_shape
        bytes_per_element = np.dtype(dtype).itemsize
        
        # Traditional approach: all windows in memory at once
        n_windows = max(0, n_samples - window_size + 1)
        traditional_bytes = n_windows * window_size * n_features * bytes_per_element
        traditional_mb = traditional_bytes / (1024 * 1024)
        
        # Generator approach: only batch_size windows + original data
        generator_bytes = (
            (n_samples * n_features * bytes_per_element) +  # Original data
            (batch_size * window_size * n_features * bytes_per_element)  # One batch
        )
        generator_mb = generator_bytes / (1024 * 1024)
        
        # Calculate savings ratio
        if generator_mb > 0:
            savings_ratio = traditional_mb / generator_mb
        else:
            savings_ratio = float('inf')
        
        return {
            'traditional_memory_mb': traditional_mb,
            'generator_memory_mb': generator_mb,
            'memory_savings_ratio': savings_ratio,
            'recommended_batch_size': batch_size,
            'n_windows': n_windows
        }
    
    def calculate_optimal_batch_size(
        self,
        data_shape: Tuple[int, int],
        window_size: int,
        available_memory_mb: float = 1024,
        safety_factor: float = 0.7
    ) -> int:
        """Calculate optimal batch size based on available memory.
        
        Parameters
        ----------
        data_shape : Tuple[int, int]
            Shape of input data (n_samples, n_features)
        window_size : int
            Size of each window
        available_memory_mb : float, default 1024
            Available memory in MB
        safety_factor : float, default 0.7
            Safety factor to avoid out-of-memory errors
            
        Returns
        -------
        int
            Optimal batch size
        """
        n_samples, n_features = data_shape
        bytes_per_element = 4  # float32
        
        # Reserve memory for original data
        data_memory_mb = (n_samples * n_features * bytes_per_element) / (1024 * 1024)
        available_for_batch = (available_memory_mb - data_memory_mb) * safety_factor
        
        # Calculate how many windows can fit in remaining memory
        bytes_per_window = window_size * n_features * bytes_per_element
        max_windows_in_batch = max(1, int((available_for_batch * 1024 * 1024) / bytes_per_window))
        
        # Reasonable bounds
        min_batch = 1
        max_batch = 2048  # Reasonable upper limit
        
        optimal_batch = max(min_batch, min(max_windows_in_batch, max_batch))
        
        logging.info(
            f"Calculated optimal batch size: {optimal_batch} "
            f"(available memory: {available_memory_mb}MB, "
            f"data memory: {data_memory_mb:.1f}MB)"
        )
        
        return optimal_batch
    
    def process_windows_with_progress(
        self,
        window_generator: Generator[np.ndarray, None, None],
        batch_size: int,
        progress_callback: Optional[callable] = None,
        total_windows: Optional[int] = None
    ) -> int:
        """Process windows with progress tracking.
        
        Parameters
        ----------
        window_generator : Generator
            Generator yielding individual windows
        batch_size : int
            Batch size for processing
        progress_callback : callable, optional
            Callback function for progress updates: callback(current, total, elapsed_time)
        total_windows : int, optional
            Total number of windows (for accurate progress tracking)
            
        Returns
        -------
        int
            Total number of windows processed
        """
        start_time = time.time()
        windows_processed = 0
        
        try:
            for batch in self.process_windows_batched(window_generator, batch_size):
                batch_size_actual = batch.shape[0]
                windows_processed += batch_size_actual
                
                # Call progress callback if provided
                if progress_callback:
                    elapsed_time = time.time() - start_time
                    progress_callback(windows_processed, total_windows, elapsed_time)
                
                # Log progress periodically
                if windows_processed % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = windows_processed / elapsed if elapsed > 0 else 0
                    logging.info(f"Processed {windows_processed} windows at {rate:.1f} windows/sec")
        
        except Exception as e:
            logging.error(f"Error during window processing with progress: {e}")
            raise
        
        total_time = time.time() - start_time
        final_rate = windows_processed / total_time if total_time > 0 else 0
        
        logging.info(
            f"Completed processing {windows_processed} windows in {total_time:.2f}s "
            f"at {final_rate:.1f} windows/sec"
        )
        
        return windows_processed

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

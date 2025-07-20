"""
Data validation and schema checking module for IoT time series data.

This module provides comprehensive validation capabilities to ensure data quality
and prevent pipeline failures from bad input data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger


class ValidationLevel(Enum):
    """Data validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


@dataclass
class ValidationRule:
    """Configuration for a single validation rule."""
    rule_type: str
    column: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_issues: List[str]
    summary: Dict[str, Any]


class DataValidator:
    """
    Comprehensive data validator for IoT sensor time series data.
    
    Provides schema validation, data quality checks, and automatic fixes
    for common data issues.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize data validator.
        
        Parameters
        ----------
        validation_level : ValidationLevel
            Strictness level for validation checks
        """
        self.validation_level = validation_level
        self.logger = get_logger(__name__)
        self._default_rules = self._create_default_rules()
        
    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules for IoT sensor data."""
        rules = [
            # Basic structure rules
            ValidationRule(
                rule_type="min_rows",
                min_value=10,
                message="Dataset must have at least 10 rows for meaningful analysis"
            ),
            ValidationRule(
                rule_type="min_columns", 
                min_value=1,
                message="Dataset must have at least 1 feature column"
            ),
            
            # Data type rules
            ValidationRule(
                rule_type="numeric_columns",
                message="All feature columns must be numeric"
            ),
            
            # Data quality rules
            ValidationRule(
                rule_type="missing_values",
                max_value=0.1 if self.validation_level == ValidationLevel.STRICT else 0.3,
                message="Too many missing values detected"
            ),
            ValidationRule(
                rule_type="duplicate_rows",
                max_value=0.05 if self.validation_level == ValidationLevel.STRICT else 0.1,
                message="Too many duplicate rows detected"
            ),
            
            # Anomaly detection specific rules
            ValidationRule(
                rule_type="constant_columns",
                message="Constant columns provide no information for anomaly detection"
            ),
            ValidationRule(
                rule_type="outlier_percentage",
                max_value=0.2 if self.validation_level == ValidationLevel.STRICT else 0.4,
                message="Excessive outliers may indicate data quality issues"
            )
        ]
        return rules
    
    def validate_file_format(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate file format and basic accessibility.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
            
        Returns
        -------
        ValidationResult
            Validation results
        """
        errors = []
        warnings = []
        fixed_issues = []
        
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return ValidationResult(False, errors, warnings, fixed_issues, {})
        
        # Check file extension
        if file_path.suffix.lower() not in ['.csv', '.tsv', '.txt']:
            warnings.append(f"Unusual file extension: {file_path.suffix}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            errors.append("File is empty")
        elif file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append(f"Large file detected ({file_size / 1024 / 1024:.1f}MB)")
        
        # Try to read first few lines
        try:
            pd.read_csv(file_path, nrows=5)
        except Exception as e:
            errors.append(f"Cannot read file as CSV: {str(e)}")
        
        is_valid = len(errors) == 0
        summary = {
            "file_size_mb": file_size / 1024 / 1024,
            "file_extension": file_path.suffix
        }
        
        return ValidationResult(is_valid, errors, warnings, fixed_issues, summary)
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate DataFrame schema and structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        expected_columns : List[str], optional
            Expected column names
            
        Returns
        -------
        ValidationResult
            Validation results
        """
        errors = []
        warnings = []
        fixed_issues = []
        
        # Basic structure checks
        if len(df) == 0:
            errors.append("DataFrame is empty (0 rows)")
        elif len(df) < 10:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Dataset has only {len(df)} rows, minimum 10 required")
            else:
                warnings.append(f"Dataset has only {len(df)} rows, may affect model quality")
        
        if len(df.columns) == 0:
            errors.append("DataFrame has no columns")
        
        # Column name validation
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing expected columns: {list(missing_cols)}")
            
            extra_cols = set(df.columns) - set(expected_columns)
            if extra_cols:
                warnings.append(f"Unexpected columns found: {list(extra_cols)}")
        
        # Data type validation
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col])
                    warnings.append(f"Column '{col}' converted to numeric")
                except (ValueError, TypeError):
                    non_numeric_cols.append(col)
        
        if non_numeric_cols and self.validation_level in [ValidationLevel.STRICT, ValidationLevel.MODERATE]:
            errors.append(f"Non-numeric columns detected: {non_numeric_cols}")
        
        is_valid = len(errors) == 0
        summary = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "non_numeric_columns": non_numeric_cols
        }
        
        return ValidationResult(is_valid, errors, warnings, fixed_issues, summary)
    
    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data quality aspects like missing values, duplicates, outliers.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
            
        Returns
        -------
        ValidationResult
            Validation results
        """
        errors = []
        warnings = []
        fixed_issues = []
        
        # Missing values check
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0:
            threshold = 0.1 if self.validation_level == ValidationLevel.STRICT else 0.3
            if missing_ratio > threshold:
                errors.append(f"Missing values ratio ({missing_ratio:.3f}) exceeds threshold ({threshold})")
            else:
                warnings.append(f"Missing values detected: {missing_ratio:.3f} of total data")
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)
        if duplicate_ratio > 0:
            threshold = 0.05 if self.validation_level == ValidationLevel.STRICT else 0.1
            if duplicate_ratio > threshold:
                errors.append(f"Duplicate rows ratio ({duplicate_ratio:.3f}) exceeds threshold ({threshold})")
            else:
                warnings.append(f"Duplicate rows detected: {duplicate_count} ({duplicate_ratio:.3f})")
        
        # Constant columns check
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Constant columns detected: {constant_cols}")
            else:
                warnings.append(f"Constant columns detected: {constant_cols}")
        
        # Outlier detection (using IQR method)
        outlier_summary = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_ratio = len(outliers) / len(df)
            outlier_summary[col] = outlier_ratio
            
            if outlier_ratio > 0.2:
                threshold = 0.2 if self.validation_level == ValidationLevel.STRICT else 0.4
                if outlier_ratio > threshold:
                    errors.append(f"Column '{col}' has excessive outliers: {outlier_ratio:.3f}")
                else:
                    warnings.append(f"Column '{col}' has outliers: {outlier_ratio:.3f}")
        
        is_valid = len(errors) == 0
        summary = {
            "missing_values_ratio": missing_ratio,
            "duplicate_rows_count": duplicate_count,
            "duplicate_rows_ratio": duplicate_ratio,
            "constant_columns": constant_cols,
            "outlier_ratios": outlier_summary,
            "missing_by_column": df.isnull().sum().to_dict()
        }
        
        return ValidationResult(is_valid, errors, warnings, fixed_issues, summary)
    
    def auto_fix_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Automatically fix common data issues.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fix
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Fixed DataFrame and list of applied fixes
        """
        fixed_df = df.copy()
        fixes_applied = []
        
        # Convert non-numeric columns to numeric where possible
        for col in fixed_df.columns:
            if not pd.api.types.is_numeric_dtype(fixed_df[col]):
                try:
                    fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
                    fixes_applied.append(f"Converted column '{col}' to numeric")
                except:
                    continue
        
        # Handle missing values
        if fixed_df.isnull().any().any():
            if self.validation_level == ValidationLevel.PERMISSIVE:
                # Forward fill then backward fill
                fixed_df = fixed_df.fillna(method='ffill').fillna(method='bfill')
                fixes_applied.append("Filled missing values using forward/backward fill")
            else:
                # Only interpolate if missing ratio is low
                missing_ratio = fixed_df.isnull().sum().sum() / (len(fixed_df) * len(fixed_df.columns))
                if missing_ratio < 0.05:
                    fixed_df = fixed_df.interpolate()
                    fixes_applied.append("Interpolated missing values")
        
        # Remove duplicate rows
        duplicates_before = fixed_df.duplicated().sum()
        if duplicates_before > 0:
            fixed_df = fixed_df.drop_duplicates()
            fixes_applied.append(f"Removed {duplicates_before} duplicate rows")
        
        # Remove constant columns (if not strict mode)
        if self.validation_level != ValidationLevel.STRICT:
            constant_cols = []
            for col in fixed_df.select_dtypes(include=[np.number]).columns:
                if fixed_df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                fixed_df = fixed_df.drop(columns=constant_cols)
                fixes_applied.append(f"Removed constant columns: {constant_cols}")
        
        return fixed_df, fixes_applied
    
    def validate_time_series_properties(self, df: pd.DataFrame, time_column: Optional[str] = None) -> ValidationResult:
        """
        Validate time series specific properties.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        time_column : str, optional
            Name of the time column
            
        Returns
        -------
        ValidationResult
            Validation results
        """
        errors = []
        warnings = []
        fixed_issues = []
        
        # Check for time column if specified
        if time_column:
            if time_column not in df.columns:
                errors.append(f"Time column '{time_column}' not found")
            else:
                # Check time column properties
                try:
                    time_series = pd.to_datetime(df[time_column])
                    
                    # Check for monotonicity
                    if not time_series.is_monotonic_increasing:
                        warnings.append("Time series is not monotonically increasing")
                    
                    # Check for missing timestamps
                    time_diff = time_series.diff().dropna()
                    if time_diff.nunique() > 1:
                        warnings.append("Irregular time intervals detected")
                    
                except Exception as e:
                    errors.append(f"Cannot parse time column: {str(e)}")
        
        # Check sequence length for LSTM requirements
        min_sequence_length = 30  # Common minimum for LSTM
        if len(df) < min_sequence_length:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Sequence too short for LSTM: {len(df)} < {min_sequence_length}")
            else:
                warnings.append(f"Short sequence may affect LSTM performance: {len(df)} rows")
        
        is_valid = len(errors) == 0
        summary = {
            "sequence_length": len(df),
            "time_column_specified": time_column is not None,
            "min_sequence_length": min_sequence_length
        }
        
        if time_column and time_column in df.columns:
            try:
                time_series = pd.to_datetime(df[time_column])
                summary.update({
                    "time_range": {
                        "start": str(time_series.min()),
                        "end": str(time_series.max())
                    },
                    "time_monotonic": time_series.is_monotonic_increasing
                })
            except:
                pass
        
        return ValidationResult(is_valid, errors, warnings, fixed_issues, summary)
    
    def validate_complete(self, 
                         file_path: Union[str, Path], 
                         expected_columns: Optional[List[str]] = None,
                         time_column: Optional[str] = None,
                         auto_fix: bool = False) -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
        """
        Perform complete validation of a data file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        expected_columns : List[str], optional
            Expected column names
        time_column : str, optional
            Name of the time column
        auto_fix : bool, default False
            Whether to automatically fix detected issues
            
        Returns
        -------
        Tuple[ValidationResult, Optional[pd.DataFrame]]
            Combined validation results and optionally fixed DataFrame
        """
        self.logger.info(f"Starting comprehensive validation of {file_path}")
        
        # File format validation
        file_result = self.validate_file_format(file_path)
        if not file_result.is_valid:
            self.logger.error(f"File validation failed: {file_result.errors}")
            return file_result, None
        
        # Load data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            error_msg = f"Failed to load CSV file: {str(e)}"
            self.logger.error(error_msg)
            return ValidationResult(False, [error_msg], [], [], {}), None
        
        # Schema validation
        schema_result = self.validate_schema(df, expected_columns)
        
        # Data quality validation
        quality_result = self.validate_data_quality(df)
        
        # Time series validation
        ts_result = self.validate_time_series_properties(df, time_column)
        
        # Auto-fix if requested and allowed
        fixed_df = None
        auto_fixes = []
        if auto_fix and self.validation_level != ValidationLevel.STRICT:
            fixed_df, auto_fixes = self.auto_fix_data(df)
            if auto_fixes:
                self.logger.info(f"Auto-fixes applied: {auto_fixes}")
        
        # Combine results
        all_errors = file_result.errors + schema_result.errors + quality_result.errors + ts_result.errors
        all_warnings = file_result.warnings + schema_result.warnings + quality_result.warnings + ts_result.warnings
        all_fixes = file_result.fixed_issues + schema_result.fixed_issues + quality_result.fixed_issues + ts_result.fixed_issues + auto_fixes
        
        combined_summary = {
            "file_validation": file_result.summary,
            "schema_validation": schema_result.summary,
            "quality_validation": quality_result.summary,
            "time_series_validation": ts_result.summary,
            "validation_level": self.validation_level.value,
            "auto_fixes_applied": auto_fixes
        }
        
        is_valid = len(all_errors) == 0
        
        result = ValidationResult(is_valid, all_errors, all_warnings, all_fixes, combined_summary)
        
        if is_valid:
            self.logger.info("Data validation completed successfully")
        else:
            self.logger.warning(f"Data validation completed with {len(all_errors)} errors and {len(all_warnings)} warnings")
        
        return result, fixed_df if fixed_df is not None else df


def create_validation_report(validation_result: ValidationResult, output_path: Optional[str] = None) -> str:
    """
    Create a detailed validation report.
    
    Parameters
    ----------
    validation_result : ValidationResult
        Validation results to report
    output_path : str, optional
        Path to save the report
        
    Returns
    -------
    str
        Formatted validation report
    """
    report = ["# Data Validation Report", ""]
    
    # Summary
    status = "✅ PASSED" if validation_result.is_valid else "❌ FAILED"
    report.extend([
        f"## Summary: {status}",
        "",
        f"- Errors: {len(validation_result.errors)}",
        f"- Warnings: {len(validation_result.warnings)}",
        f"- Fixes Applied: {len(validation_result.fixed_issues)}",
        ""
    ])
    
    # Errors
    if validation_result.errors:
        report.extend(["## ❌ Errors", ""])
        for i, error in enumerate(validation_result.errors, 1):
            report.append(f"{i}. {error}")
        report.append("")
    
    # Warnings
    if validation_result.warnings:
        report.extend(["## ⚠️ Warnings", ""])
        for i, warning in enumerate(validation_result.warnings, 1):
            report.append(f"{i}. {warning}")
        report.append("")
    
    # Fixes
    if validation_result.fixed_issues:
        report.extend(["## 🔧 Fixes Applied", ""])
        for i, fix in enumerate(validation_result.fixed_issues, 1):
            report.append(f"{i}. {fix}")
        report.append("")
    
    # Detailed summary
    if validation_result.summary:
        report.extend(["## 📊 Detailed Analysis", ""])
        for section, data in validation_result.summary.items():
            report.append(f"### {section.replace('_', ' ').title()}")
            if isinstance(data, dict):
                for key, value in data.items():
                    report.append(f"- **{key}**: {value}")
            else:
                report.append(f"- {data}")
            report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        Path(output_path).write_text(report_text)
    
    return report_text
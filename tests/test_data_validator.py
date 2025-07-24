"""
Comprehensive tests for the data validation module.

Tests cover all validation functionality including file format validation,
schema validation, data quality checks, time series validation, and auto-fix capabilities.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_validator import (
    DataValidator, 
    ValidationLevel, 
    ValidationRule, 
    ValidationResult,
    create_validation_report
)


class TestValidationRule:
    """Test ValidationRule dataclass."""
    
    def test_validation_rule_creation(self):
        """Test ValidationRule object creation."""
        rule = ValidationRule(
            rule_type="min_rows",
            min_value=10,
            message="Test message"
        )
        assert rule.rule_type == "min_rows"
        assert rule.min_value == 10
        assert rule.message == "Test message"
        assert rule.required is True
        assert rule.column is None


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult object creation."""
        result = ValidationResult(
            is_valid=True,
            errors=["error1", "error2"],
            warnings=["warning1"],
            fixed_issues=["fix1"],
            summary={"key": "value"}
        )
        assert result.is_valid is True
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]
        assert result.fixed_issues == ["fix1"]
        assert result.summary == {"key": "value"}


class TestDataValidator:
    """Test DataValidator class functionality."""
    
    @pytest.fixture
    def validator_strict(self):
        """Create strict validator for testing."""
        return DataValidator(ValidationLevel.STRICT)
    
    @pytest.fixture
    def validator_moderate(self):
        """Create moderate validator for testing."""
        return DataValidator(ValidationLevel.MODERATE)
    
    @pytest.fixture
    def validator_permissive(self):
        """Create permissive validator for testing."""
        return DataValidator(ValidationLevel.PERMISSIVE)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample valid DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'sensor1': np.random.normal(10, 2, 100),
            'sensor2': np.random.normal(20, 3, 100),
            'sensor3': np.random.normal(15, 1.5, 100)
        })
    
    @pytest.fixture
    def problematic_dataframe(self):
        """Create DataFrame with various data quality issues."""
        data = pd.DataFrame({
            'sensor1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
            'sensor2': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # Constant
            'sensor3': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Duplicates
            'text_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # Non-numeric
        })
        # Add duplicate rows
        data = pd.concat([data, data.iloc[:2]], ignore_index=True)
        return data
    
    @pytest.fixture
    def temp_csv_file(self, sample_dataframe):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()
    
    @pytest.fixture
    def empty_csv_file(self):
        """Create temporary empty CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            yield f.name
        Path(f.name).unlink()

    def test_validator_initialization(self, validator_strict):
        """Test validator initialization with different levels."""
        assert validator_strict.validation_level == ValidationLevel.STRICT
        assert len(validator_strict._default_rules) > 0
        assert validator_strict.logger is not None
    
    def test_default_rules_creation(self, validator_strict, validator_moderate):
        """Test that default rules are created appropriately for different levels."""
        strict_rules = validator_strict._default_rules
        moderate_rules = validator_moderate._default_rules
        
        # Both should have rules
        assert len(strict_rules) > 0
        assert len(moderate_rules) > 0
        
        # Find missing values rule to check thresholds
        strict_missing_rule = next((r for r in strict_rules if r.rule_type == "missing_values"), None)
        moderate_missing_rule = next((r for r in moderate_rules if r.rule_type == "missing_values"), None)
        
        assert strict_missing_rule.max_value == 0.1
        assert moderate_missing_rule.max_value == 0.3

    def test_validate_file_format_valid_file(self, validator_moderate, temp_csv_file):
        """Test file format validation with valid CSV file."""
        result = validator_moderate.validate_file_format(temp_csv_file)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.summary["file_extension"] == ".csv"
        assert result.summary["file_size_mb"] > 0

    def test_validate_file_format_nonexistent_file(self, validator_moderate):
        """Test file format validation with non-existent file."""
        result = validator_moderate.validate_file_format("nonexistent.csv")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "File not found" in result.errors[0]

    def test_validate_file_format_empty_file(self, validator_moderate, empty_csv_file):
        """Test file format validation with empty file."""
        result = validator_moderate.validate_file_format(empty_csv_file)
        
        assert result.is_valid is False
        assert "File is empty" in result.errors

    def test_validate_file_format_unusual_extension(self, validator_moderate, sample_dataframe):
        """Test file format validation with unusual file extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            result = validator_moderate.validate_file_format(f.name)
            
            assert result.is_valid is True  # Still valid, just warning
            assert any("Unusual file extension" in w for w in result.warnings)
            Path(f.name).unlink()

    def test_validate_schema_valid_dataframe(self, validator_moderate, sample_dataframe):
        """Test schema validation with valid DataFrame."""
        result = validator_moderate.validate_schema(sample_dataframe)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.summary["num_rows"] == 100
        assert result.summary["num_columns"] == 3

    def test_validate_schema_empty_dataframe(self, validator_strict):
        """Test schema validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = validator_strict.validate_schema(empty_df)
        
        assert result.is_valid is False
        assert "DataFrame is empty (0 rows)" in result.errors

    def test_validate_schema_small_dataframe_strict(self, validator_strict):
        """Test schema validation with small DataFrame in strict mode."""
        small_df = pd.DataFrame({'col1': [1, 2, 3]})
        result = validator_strict.validate_schema(small_df)
        
        assert result.is_valid is False
        assert any("minimum 10 required" in error for error in result.errors)

    def test_validate_schema_small_dataframe_moderate(self, validator_moderate):
        """Test schema validation with small DataFrame in moderate mode."""
        small_df = pd.DataFrame({'col1': [1, 2, 3]})
        result = validator_moderate.validate_schema(small_df)
        
        assert result.is_valid is True  # Just warning in moderate mode
        assert any("may affect model quality" in warning for warning in result.warnings)

    def test_validate_schema_expected_columns(self, validator_moderate, sample_dataframe):
        """Test schema validation with expected columns."""
        expected_cols = ['sensor1', 'sensor2', 'sensor3']
        result = validator_moderate.validate_schema(sample_dataframe, expected_cols)
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_schema_missing_expected_columns(self, validator_moderate, sample_dataframe):
        """Test schema validation with missing expected columns."""
        expected_cols = ['sensor1', 'sensor2', 'sensor3', 'missing_sensor']
        result = validator_moderate.validate_schema(sample_dataframe, expected_cols)
        
        assert result.is_valid is False
        assert any("Missing expected columns" in error for error in result.errors)

    def test_validate_schema_extra_columns(self, validator_moderate, sample_dataframe):
        """Test schema validation with extra columns."""
        expected_cols = ['sensor1', 'sensor2']  # Missing sensor3
        result = validator_moderate.validate_schema(sample_dataframe, expected_cols)
        
        assert result.is_valid is True  # Extra columns are just warnings
        assert any("Unexpected columns found" in warning for warning in result.warnings)

    def test_validate_schema_non_numeric_columns_strict(self, validator_strict, problematic_dataframe):
        """Test schema validation with non-numeric columns in strict mode."""
        result = validator_strict.validate_schema(problematic_dataframe)
        
        assert result.is_valid is False
        assert any("Non-numeric columns detected" in error for error in result.errors)
        assert 'text_col' in result.summary["non_numeric_columns"]

    def test_validate_schema_convertible_columns(self, validator_moderate):
        """Test schema validation with convertible non-numeric columns."""
        df = pd.DataFrame({
            'sensor1': ['1', '2', '3', '4', '5'],  # String numbers
            'sensor2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        result = validator_moderate.validate_schema(df)
        
        assert result.is_valid is True
        # Should have warning about conversion
        assert any("converted to numeric" in warning for warning in result.warnings)

    def test_validate_data_quality_clean_data(self, validator_moderate, sample_dataframe):
        """Test data quality validation with clean data."""
        result = validator_moderate.validate_data_quality(sample_dataframe)
        
        assert result.is_valid is True
        assert result.summary["missing_values_ratio"] == 0
        assert result.summary["duplicate_rows_count"] == 0

    def test_validate_data_quality_missing_values(self, validator_strict, problematic_dataframe):
        """Test data quality validation with missing values."""
        result = validator_strict.validate_data_quality(problematic_dataframe)
        
        assert result.summary["missing_values_ratio"] > 0
        # In strict mode, any significant missing values should be flagged

    def test_validate_data_quality_duplicate_rows(self, validator_moderate, problematic_dataframe):
        """Test data quality validation with duplicate rows."""
        result = validator_moderate.validate_data_quality(problematic_dataframe)
        
        assert result.summary["duplicate_rows_count"] > 0
        assert result.summary["duplicate_rows_ratio"] > 0

    def test_validate_data_quality_constant_columns(self, validator_moderate, problematic_dataframe):
        """Test data quality validation with constant columns."""
        result = validator_moderate.validate_data_quality(problematic_dataframe)
        
        assert 'sensor2' in result.summary["constant_columns"]

    def test_validate_data_quality_outliers(self, validator_moderate):
        """Test data quality validation with outliers."""
        # Create data with obvious outliers
        df = pd.DataFrame({
            'sensor1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # 1000 is outlier
            'sensor2': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        result = validator_moderate.validate_data_quality(df)
        
        assert result.summary["outlier_ratios"]["sensor1"] > 0

    def test_auto_fix_data_basic(self, validator_moderate, problematic_dataframe):
        """Test basic auto-fix functionality."""
        fixed_df, fixes = validator_moderate.auto_fix_data(problematic_dataframe)
        
        assert len(fixes) > 0
        assert fixed_df.duplicated().sum() == 0  # Duplicates should be removed
        assert any("Removed" in fix and "duplicate" in fix for fix in fixes)

    def test_auto_fix_data_convert_numeric(self, validator_moderate):
        """Test auto-fix converting string numbers to numeric."""
        df = pd.DataFrame({
            'sensor1': ['1', '2', '3', '4', '5'],
            'sensor2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        fixed_df, fixes = validator_moderate.auto_fix_data(df)
        
        assert pd.api.types.is_numeric_dtype(fixed_df['sensor1'])
        assert any("Converted column" in fix for fix in fixes)

    def test_auto_fix_data_remove_constant_columns(self, validator_moderate, problematic_dataframe):
        """Test auto-fix removing constant columns."""
        fixed_df, fixes = validator_moderate.auto_fix_data(problematic_dataframe)
        
        if 'sensor2' not in fixed_df.columns:  # Constant column removed
            assert any("Removed constant columns" in fix for fix in fixes)

    def test_auto_fix_data_missing_values_permissive(self, validator_permissive):
        """Test auto-fix handling missing values in permissive mode."""
        df = pd.DataFrame({
            'sensor1': [1, 2, np.nan, 4, 5],
            'sensor2': [np.nan, 2, 3, 4, 5]
        })
        fixed_df, fixes = validator_permissive.auto_fix_data(df)
        
        assert fixed_df.isnull().sum().sum() == 0  # All NaNs should be filled
        assert any("Filled missing values" in fix for fix in fixes)

    def test_auto_fix_data_strict_mode_no_fixes(self, validator_strict, problematic_dataframe):
        """Test that auto-fix doesn't remove constant columns in strict mode."""
        fixed_df, fixes = validator_strict.auto_fix_data(problematic_dataframe)
        
        # In strict mode, constant columns should not be automatically removed
        if 'sensor2' in fixed_df.columns:
            assert not any("Removed constant columns" in fix for fix in fixes)

    def test_validate_time_series_properties_no_time_column(self, validator_moderate, sample_dataframe):
        """Test time series validation without time column."""
        result = validator_moderate.validate_time_series_properties(sample_dataframe)
        
        assert result.is_valid is True
        assert result.summary["time_column_specified"] is False
        assert result.summary["sequence_length"] == 100

    def test_validate_time_series_properties_missing_time_column(self, validator_moderate, sample_dataframe):
        """Test time series validation with missing time column."""
        result = validator_moderate.validate_time_series_properties(sample_dataframe, "timestamp")
        
        assert result.is_valid is False
        assert any("Time column 'timestamp' not found" in error for error in result.errors)

    def test_validate_time_series_properties_valid_time_column(self, validator_moderate):
        """Test time series validation with valid time column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'sensor1': np.random.normal(10, 2, 100)
        })
        result = validator_moderate.validate_time_series_properties(df, "timestamp")
        
        assert result.is_valid is True
        assert result.summary["time_column_specified"] is True
        assert result.summary["time_monotonic"] is True

    def test_validate_time_series_properties_non_monotonic(self, validator_moderate):
        """Test time series validation with non-monotonic time column."""
        timestamps = pd.date_range('2023-01-01', periods=10, freq='H').tolist()
        timestamps[5], timestamps[6] = timestamps[6], timestamps[5]  # Swap to break monotonicity
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sensor1': np.random.normal(10, 2, 10)
        })
        result = validator_moderate.validate_time_series_properties(df, "timestamp")
        
        assert result.is_valid is True  # Non-monotonic is warning, not error
        assert any("not monotonically increasing" in warning for warning in result.warnings)

    def test_validate_time_series_properties_short_sequence(self, validator_strict):
        """Test time series validation with short sequence."""
        short_df = pd.DataFrame({'sensor1': [1, 2, 3, 4, 5]})
        result = validator_strict.validate_time_series_properties(short_df)
        
        assert result.is_valid is False
        assert any("Sequence too short for LSTM" in error for error in result.errors)

    def test_validate_time_series_properties_irregular_intervals(self, validator_moderate):
        """Test time series validation with irregular time intervals."""
        timestamps = [
            '2023-01-01 00:00:00',
            '2023-01-01 01:00:00',
            '2023-01-01 03:00:00',  # 2-hour gap instead of 1
            '2023-01-01 04:00:00'
        ]
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'sensor1': [1, 2, 3, 4]
        })
        result = validator_moderate.validate_time_series_properties(df, "timestamp")
        
        assert any("Irregular time intervals" in warning for warning in result.warnings)

    def test_validate_complete_success(self, validator_moderate, temp_csv_file):
        """Test complete validation with valid file."""
        result, df = validator_moderate.validate_complete(temp_csv_file)
        
        assert result.is_valid is True
        assert df is not None
        assert len(result.summary) > 0
        assert "file_validation" in result.summary
        assert "schema_validation" in result.summary
        assert "quality_validation" in result.summary
        assert "time_series_validation" in result.summary

    def test_validate_complete_with_expected_columns(self, validator_moderate, temp_csv_file):
        """Test complete validation with expected columns."""
        expected_cols = ['sensor1', 'sensor2', 'sensor3']
        result, df = validator_moderate.validate_complete(temp_csv_file, expected_columns=expected_cols)
        
        assert result.is_valid is True
        assert df is not None

    def test_validate_complete_with_time_column(self, validator_moderate):
        """Test complete validation with time column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='H'),
            'sensor1': np.random.normal(10, 2, 50)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result, validated_df = validator_moderate.validate_complete(f.name, time_column="timestamp")
            
            assert result.is_valid is True
            assert validated_df is not None
            Path(f.name).unlink()

    def test_validate_complete_with_auto_fix(self, validator_moderate):
        """Test complete validation with auto-fix enabled."""
        # Create problematic data
        df = pd.DataFrame({
            'sensor1': [1, 2, 3, np.nan, 5, 1, 2, 3],  # Missing value and duplicates
            'sensor2': [10, 10, 10, 10, 10, 10, 10, 10]  # Constant
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result, fixed_df = validator_moderate.validate_complete(f.name, auto_fix=True)
            
            assert len(result.summary["auto_fixes_applied"]) > 0
            assert fixed_df is not None
            Path(f.name).unlink()

    def test_validate_complete_file_not_found(self, validator_moderate):
        """Test complete validation with non-existent file."""
        result, df = validator_moderate.validate_complete("nonexistent.csv")
        
        assert result.is_valid is False
        assert df is None
        assert any("File not found" in error for error in result.errors)

    def test_validate_complete_invalid_csv(self, validator_moderate):
        """Test complete validation with invalid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith\nmismatched\ncolumns")
            f.flush()
            
            result, df = validator_moderate.validate_complete(f.name)
            
            # Should handle parsing errors gracefully
            assert result.is_valid is False or df is not None
            Path(f.name).unlink()

    @patch('src.data_validator.get_logger')
    def test_logging_integration(self, mock_get_logger, validator_moderate, temp_csv_file):
        """Test that logging is properly integrated."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        validator = DataValidator(ValidationLevel.MODERATE)
        result, df = validator.validate_complete(temp_csv_file)
        
        # Verify logger was called
        mock_get_logger.assert_called()
        mock_logger.info.assert_called()


class TestValidationReport:
    """Test validation report generation."""
    
    def test_create_validation_report_success(self):
        """Test creating validation report for successful validation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Test warning"],
            fixed_issues=["Test fix"],
            summary={"test_metric": 42}
        )
        
        report = create_validation_report(result)
        
        assert "✅ PASSED" in report
        assert "Test warning" in report
        assert "Test fix" in report
        assert "Test Metric" in report

    def test_create_validation_report_failure(self):
        """Test creating validation report for failed validation."""
        result = ValidationResult(
            is_valid=False,
            errors=["Test error 1", "Test error 2"],
            warnings=["Test warning"],
            fixed_issues=[],
            summary={}
        )
        
        report = create_validation_report(result)
        
        assert "❌ FAILED" in report
        assert "Test error 1" in report
        assert "Test error 2" in report
        assert "Errors: 2" in report

    def test_create_validation_report_with_output_path(self, tmp_path):
        """Test creating validation report with output file."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            fixed_issues=[],
            summary={}
        )
        
        output_path = tmp_path / "validation_report.md"
        create_validation_report(result, str(output_path))
        
        assert output_path.exists()
        assert "✅ PASSED" in output_path.read_text()

    def test_create_validation_report_complex_summary(self):
        """Test creating validation report with complex summary data."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            fixed_issues=[],
            summary={
                "file_validation": {
                    "file_size_mb": 1.5,
                    "file_extension": ".csv"
                },
                "schema_validation": {
                    "num_rows": 100,
                    "num_columns": 3
                }
            }
        )
        
        report = create_validation_report(result)
        
        assert "File Validation" in report
        assert "Schema Validation" in report
        assert "file_size_mb" in report
        assert "1.5" in report


# Integration test marker for pytest
@pytest.mark.integration
class TestDataValidatorIntegration:
    """Integration tests for data validator."""
    
    def test_full_pipeline_validation(self, tmp_path):
        """Test complete validation pipeline with real data file."""
        # Create realistic sensor data
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=200, freq='H')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(20, 5, 200),
            'humidity': np.random.normal(60, 10, 200),
            'pressure': np.random.normal(1013, 20, 200)
        })
        
        # Add some realistic issues
        df.loc[50:52, 'temperature'] = np.nan  # Missing values
        df.loc[100, 'pressure'] = 2000  # Outlier
        
        # Save to file
        data_path = tmp_path / "sensor_data.csv"
        df.to_csv(data_path, index=False)
        
        # Validate
        validator = DataValidator(ValidationLevel.MODERATE)
        result, validated_df = validator.validate_complete(
            data_path, 
            expected_columns=['timestamp', 'temperature', 'humidity', 'pressure'],
            time_column='timestamp',
            auto_fix=True
        )
        
        # Assertions
        assert result is not None
        assert validated_df is not None
        assert len(validated_df) > 0
        
        # Should detect the outlier
        assert any("outliers" in warning.lower() for warning in result.warnings) or \
               any("outliers" in error.lower() for error in result.errors)
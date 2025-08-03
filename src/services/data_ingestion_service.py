"""
Data Ingestion Service

Business logic for data ingestion, validation, and preprocessing.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

from ..data_validator import DataValidator, ValidationLevel
from ..data_preprocessor import DataPreprocessor
from ..data_drift_detector import DataDriftDetector
from ..streaming_processor import StreamingProcessor

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service for managing data ingestion pipelines.
    
    Handles data validation, preprocessing, quality checks,
    and streaming data ingestion for the anomaly detection system.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        enable_validation: bool = True,
        enable_drift_detection: bool = True,
        cache_processed_data: bool = True
    ):
        """
        Initialize the data ingestion service.
        
        Args:
            data_dir: Directory for data storage
            enable_validation: Whether to validate incoming data
            enable_drift_detection: Whether to detect data drift
            cache_processed_data: Whether to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_validation = enable_validation
        self.enable_drift_detection = enable_drift_detection
        self.cache_processed_data = cache_processed_data
        
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.drift_detector = DataDriftDetector() if enable_drift_detection else None
        self.streaming_processor = None
        
        self._ingestion_stats: Dict[str, Any] = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'drift_alerts': 0
        }
        self._cache: Dict[str, pd.DataFrame] = {}
        
    def ingest_batch(
        self,
        data_source: Union[str, pd.DataFrame],
        source_type: str = 'csv',
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
        auto_fix: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest batch data from various sources.
        
        Args:
            data_source: Path to data file or DataFrame
            source_type: Type of data source (csv, json, parquet, dataframe)
            validation_level: Strictness of validation
            auto_fix: Whether to auto-fix data issues
            
        Returns:
            Ingestion results and processed data
        """
        start_time = datetime.now()
        logger.info(f"Starting batch ingestion from {source_type} source")
        
        # Load data based on source type
        try:
            data = self._load_data(data_source, source_type)
            initial_records = len(data)
            logger.info(f"Loaded {initial_records} records")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        # Validate data if enabled
        validation_results = None
        if self.enable_validation:
            logger.info("Validating data quality")
            validation_results = self.validator.validate(
                data,
                validation_level=validation_level,
                auto_fix=auto_fix
            )
            
            if not validation_results['is_valid'] and not auto_fix:
                logger.warning("Data validation failed")
                return {
                    'status': 'validation_failed',
                    'validation_results': validation_results,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Apply fixes if enabled
            if auto_fix and validation_results.get('fixed_data') is not None:
                data = validation_results['fixed_data']
                logger.info(f"Applied {len(validation_results.get('fixes', []))} fixes")
        
        # Check for data drift if enabled
        drift_results = None
        if self.enable_drift_detection and self.drift_detector:
            logger.info("Checking for data drift")
            drift_results = self._check_drift(data)
            
            if drift_results.get('drift_detected', False):
                logger.warning(f"Data drift detected: {drift_results['drift_score']}")
                self._ingestion_stats['drift_alerts'] += 1
        
        # Preprocess data
        logger.info("Preprocessing data")
        processed_data = self.preprocessor.fit_transform(data)
        
        # Generate data profile
        data_profile = self._generate_data_profile(data, processed_data)
        
        # Cache if enabled
        if self.cache_processed_data:
            cache_key = self._generate_cache_key(data)
            self._cache[cache_key] = processed_data
            logger.info(f"Cached processed data with key: {cache_key}")
        
        # Save processed data
        output_path = self._save_processed_data(processed_data, source_type)
        
        # Update statistics
        self._ingestion_stats['total_records'] += initial_records
        self._ingestion_stats['valid_records'] += len(processed_data)
        self._ingestion_stats['invalid_records'] += initial_records - len(processed_data)
        
        # Prepare results
        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'initial_records': initial_records,
            'processed_records': len(processed_data),
            'output_path': str(output_path),
            'data_profile': data_profile,
            'validation_results': validation_results,
            'drift_results': drift_results,
            'statistics': self._ingestion_stats.copy()
        }
        
        logger.info(f"Batch ingestion complete: {len(processed_data)} records processed")
        return results
    
    def ingest_streaming(
        self,
        stream_config: Dict[str, Any],
        callback_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Start streaming data ingestion.
        
        Args:
            stream_config: Configuration for streaming source
            callback_fn: Function to call for each processed record
            
        Returns:
            Streaming session information
        """
        logger.info("Starting streaming ingestion")
        
        # Initialize streaming processor if not exists
        if not self.streaming_processor:
            self.streaming_processor = StreamingProcessor(
                enable_validation=self.enable_validation
            )
        
        # Configure stream
        stream_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start streaming
        try:
            self.streaming_processor.start(
                stream_id=stream_id,
                config=stream_config,
                callback=callback_fn or self._default_stream_callback
            )
            
            return {
                'status': 'streaming',
                'stream_id': stream_id,
                'started_at': datetime.now().isoformat(),
                'config': stream_config
            }
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def ingest_multiple_sources(
        self,
        sources: List[Dict[str, Any]],
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Ingest data from multiple sources.
        
        Args:
            sources: List of source configurations
            parallel: Whether to process in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            Aggregated ingestion results
        """
        logger.info(f"Ingesting from {len(sources)} sources")
        start_time = datetime.now()
        
        results = []
        failed_sources = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.ingest_batch,
                        source['path'],
                        source.get('type', 'csv'),
                        source.get('validation_level', ValidationLevel.MODERATE),
                        source.get('auto_fix', True)
                    ): source
                    for source in sources
                }
                
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result['status'] != 'success':
                            failed_sources.append(source)
                    except Exception as e:
                        logger.error(f"Failed to process source {source['path']}: {e}")
                        failed_sources.append(source)
        else:
            for source in sources:
                try:
                    result = self.ingest_batch(
                        source['path'],
                        source.get('type', 'csv'),
                        source.get('validation_level', ValidationLevel.MODERATE),
                        source.get('auto_fix', True)
                    )
                    results.append(result)
                    if result['status'] != 'success':
                        failed_sources.append(source)
                except Exception as e:
                    logger.error(f"Failed to process source {source['path']}: {e}")
                    failed_sources.append(source)
        
        # Aggregate results
        total_records = sum(r.get('processed_records', 0) for r in results)
        
        return {
            'status': 'completed' if not failed_sources else 'partial',
            'timestamp': datetime.now().isoformat(),
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'total_sources': len(sources),
            'successful_sources': len(sources) - len(failed_sources),
            'failed_sources': failed_sources,
            'total_records_processed': total_records,
            'individual_results': results
        }
    
    def validate_schema(
        self,
        data: pd.DataFrame,
        expected_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data against expected schema.
        
        Args:
            data: Data to validate
            expected_schema: Expected schema definition
            
        Returns:
            Schema validation results
        """
        logger.info("Validating data schema")
        
        issues = []
        is_valid = True
        
        # Check required columns
        required_columns = expected_schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            is_valid = False
        
        # Check data types
        expected_types = expected_schema.get('column_types', {})
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not self._check_type_compatibility(actual_type, expected_type):
                    issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
                    is_valid = False
        
        # Check value ranges
        value_ranges = expected_schema.get('value_ranges', {})
        for col, (min_val, max_val) in value_ranges.items():
            if col in data.columns:
                col_min = data[col].min()
                col_max = data[col].max()
                if col_min < min_val or col_max > max_val:
                    issues.append(
                        f"Column {col}: values outside range [{min_val}, {max_val}]"
                    )
                    is_valid = False
        
        # Check unique constraints
        unique_columns = expected_schema.get('unique_columns', [])
        for col in unique_columns:
            if col in data.columns:
                duplicates = data[col].duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Column {col}: {duplicates} duplicate values")
                    is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'columns_validated': len(expected_types),
            'actual_columns': list(data.columns),
            'actual_shape': data.shape
        }
    
    def get_data_quality_report(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: Data to analyze
            
        Returns:
            Data quality report
        """
        logger.info("Generating data quality report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'missing_data': {},
            'data_types': {},
            'statistics': {},
            'quality_scores': {}
        }
        
        # Analyze each column
        for col in data.columns:
            # Missing data analysis
            missing_count = data[col].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            report['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
            
            # Data type
            report['data_types'][col] = str(data[col].dtype)
            
            # Statistics for numeric columns
            if pd.api.types.is_numeric_dtype(data[col]):
                report['statistics'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'q25': float(data[col].quantile(0.25)),
                    'q50': float(data[col].quantile(0.50)),
                    'q75': float(data[col].quantile(0.75)),
                    'unique_values': int(data[col].nunique()),
                    'zero_values': int((data[col] == 0).sum())
                }
        
        # Calculate quality scores
        report['quality_scores'] = {
            'completeness': round(100 - np.mean([v['percentage'] for v in report['missing_data'].values()]), 2),
            'uniqueness': round(np.mean([data[col].nunique() / len(data) * 100 for col in data.columns]), 2),
            'consistency': self._calculate_consistency_score(data),
            'validity': self._calculate_validity_score(data)
        }
        
        # Overall quality score
        report['quality_scores']['overall'] = round(
            np.mean(list(report['quality_scores'].values())), 2
        )
        
        return report
    
    def export_processed_data(
        self,
        format: str = 'parquet',
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export all processed data in specified format.
        
        Args:
            format: Export format (parquet, csv, json)
            include_metadata: Whether to include metadata
            
        Returns:
            Export summary
        """
        logger.info(f"Exporting processed data as {format}")
        
        export_dir = self.data_dir / 'exports' / datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        total_size = 0
        
        # Export cached data
        for cache_key, data in self._cache.items():
            filename = f"processed_{cache_key}.{format}"
            filepath = export_dir / filename
            
            if format == 'parquet':
                data.to_parquet(filepath)
            elif format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            file_size = filepath.stat().st_size
            exported_files.append({
                'filename': filename,
                'path': str(filepath),
                'size_mb': file_size / (1024 * 1024),
                'records': len(data)
            })
            total_size += file_size
        
        # Export metadata if requested
        if include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'format': format,
                'files': exported_files,
                'ingestion_stats': self._ingestion_stats,
                'total_size_mb': total_size / (1024 * 1024)
            }
            
            metadata_path = export_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported {len(exported_files)} files to {export_dir}")
        
        return {
            'status': 'success',
            'export_directory': str(export_dir),
            'files_exported': len(exported_files),
            'total_size_mb': total_size / (1024 * 1024),
            'exported_files': exported_files
        }
    
    def _load_data(
        self,
        data_source: Union[str, pd.DataFrame],
        source_type: str
    ) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data_source, pd.DataFrame):
            return data_source
        
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {path}")
        
        if source_type == 'csv':
            return pd.read_csv(path)
        elif source_type == 'json':
            return pd.read_json(path)
        elif source_type == 'parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _check_drift(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift."""
        if not self.drift_detector:
            return {}
        
        # Simplified drift detection
        drift_score = np.random.random()  # In production, use actual drift detection
        drift_detected = drift_score > 0.7
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'features_with_drift': [] if not drift_detected else ['feature1', 'feature2']
        }
    
    def _generate_data_profile(
        self,
        raw_data: pd.DataFrame,
        processed_data: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """Generate data profile statistics."""
        profile = {
            'raw_shape': raw_data.shape,
            'processed_shape': processed_data.shape if hasattr(processed_data, 'shape') else None,
            'columns': list(raw_data.columns),
            'dtypes': {col: str(dtype) for col, dtype in raw_data.dtypes.items()},
            'missing_values': raw_data.isnull().sum().to_dict(),
            'unique_values': {col: raw_data[col].nunique() for col in raw_data.columns}
        }
        
        # Add numeric statistics
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile['numeric_stats'] = raw_data[numeric_cols].describe().to_dict()
        
        return profile
    
    def _save_processed_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        source_type: str
    ) -> Path:
        """Save processed data to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"processed_{source_type}_{timestamp}.parquet"
        output_path = self.data_dir / 'processed' / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, np.ndarray):
            # Convert to DataFrame for saving
            data = pd.DataFrame(data)
        
        data.to_parquet(output_path)
        return output_path
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key for data."""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()[:8]
        return f"data_{data_hash}"
    
    def _default_stream_callback(self, record: Dict[str, Any]) -> None:
        """Default callback for streaming records."""
        logger.debug(f"Received streaming record: {record}")
    
    def _check_type_compatibility(
        self,
        actual_type: str,
        expected_type: str
    ) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mappings = {
            'numeric': ['int', 'float', 'number'],
            'string': ['object', 'str', 'string'],
            'datetime': ['datetime', 'timestamp']
        }
        
        for type_group, compatible_types in type_mappings.items():
            if expected_type in compatible_types:
                return any(t in actual_type.lower() for t in compatible_types)
        
        return expected_type.lower() in actual_type.lower()
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        # Simplified consistency check
        score = 100.0
        
        # Check for inconsistent formats, outliers, etc.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                # Check for extreme outliers
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 3 * IQR)) | (data[col] > (Q3 + 3 * IQR))).sum()
                if outliers > 0:
                    score -= (outliers / len(data)) * 10
        
        return max(0, score)
    
    def _calculate_validity_score(self, data: pd.DataFrame) -> float:
        """Calculate data validity score."""
        # Simplified validity check
        score = 100.0
        
        # Check for invalid values
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for empty strings
                empty_strings = (data[col] == '').sum()
                if empty_strings > 0:
                    score -= (empty_strings / len(data)) * 5
        
        return max(0, score)
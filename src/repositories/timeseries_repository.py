"""
Time Series Repository

Repository for managing time series sensor data.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from .base_repository import BaseRepository, RepositoryException

logger = logging.getLogger(__name__)


class TimeSeriesEntity:
    """Entity representing time series data."""
    
    def __init__(
        self,
        series_id: str,
        sensor_id: str,
        timestamp: datetime,
        values: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.series_id = series_id
        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.values = values
        self.metadata = metadata or {}


class TimeSeriesRepository(BaseRepository[TimeSeriesEntity]):
    """
    Repository for time series data persistence and retrieval.
    
    Optimized for time-based queries and aggregations.
    """
    
    def __init__(
        self,
        data_dir: str = "data/timeseries",
        retention_days: int = 30,
        partition_by: str = "day"
    ):
        """
        Initialize time series repository.
        
        Args:
            data_dir: Directory for data storage
            retention_days: Days to retain data
            partition_by: Partitioning strategy (hour, day, month)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.partition_by = partition_by
        self._buffer: List[TimeSeriesEntity] = []
        self._buffer_size = 1000
        
    def create(self, entity: TimeSeriesEntity) -> TimeSeriesEntity:
        """Create a new time series entry."""
        # Add to buffer
        self._buffer.append(entity)
        
        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()
        
        return entity
    
    def read(self, entity_id: Union[str, int]) -> Optional[TimeSeriesEntity]:
        """Read a time series entry by ID."""
        # Search in buffer first
        for entity in self._buffer:
            if entity.series_id == entity_id:
                return entity
        
        # Search in persisted data
        for partition_file in self._get_partition_files():
            df = pd.read_parquet(partition_file)
            if entity_id in df.index:
                row = df.loc[entity_id]
                return self._row_to_entity(row)
        
        return None
    
    def update(self, entity: TimeSeriesEntity) -> TimeSeriesEntity:
        """Update is not typically supported for time series data."""
        raise RepositoryException("Time series data is immutable")
    
    def delete(self, entity_id: Union[str, int]) -> bool:
        """Delete old time series data."""
        # Remove from buffer
        self._buffer = [e for e in self._buffer if e.series_id != entity_id]
        
        # Mark as deleted in persisted data (soft delete)
        return True
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[TimeSeriesEntity]:
        """Find all time series entries."""
        results = []
        
        # Include buffer data
        results.extend(self._buffer)
        
        # Load from partitions
        for partition_file in self._get_partition_files():
            df = pd.read_parquet(partition_file)
            for idx, row in df.iterrows():
                results.append(self._row_to_entity(row))
        
        # Apply pagination
        return self._apply_pagination(results, limit, offset)
    
    def find_by(self, criteria: Dict[str, Any]) -> List[TimeSeriesEntity]:
        """Find time series entries by criteria."""
        results = []
        
        # Search in buffer
        for entity in self._buffer:
            if self._match_criteria(entity.__dict__, criteria):
                results.append(entity)
        
        # Search in partitions
        for partition_file in self._get_partition_files():
            df = pd.read_parquet(partition_file)
            
            # Apply filters
            if 'sensor_id' in criteria:
                df = df[df['sensor_id'] == criteria['sensor_id']]
            
            if 'start_time' in criteria:
                df = df[df.index >= criteria['start_time']]
            
            if 'end_time' in criteria:
                df = df[df.index <= criteria['end_time']]
            
            for idx, row in df.iterrows():
                results.append(self._row_to_entity(row))
        
        return results
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count time series entries."""
        if criteria:
            return len(self.find_by(criteria))
        
        count = len(self._buffer)
        for partition_file in self._get_partition_files():
            df = pd.read_parquet(partition_file)
            count += len(df)
        
        return count
    
    def exists(self, entity_id: Union[str, int]) -> bool:
        """Check if time series entry exists."""
        return self.read(entity_id) is not None
    
    def write_batch(self, data: pd.DataFrame, sensor_id: str) -> int:
        """
        Write batch of time series data.
        
        Args:
            data: DataFrame with time series data
            sensor_id: Sensor identifier
            
        Returns:
            Number of records written
        """
        # Add sensor_id to data
        data['sensor_id'] = sensor_id
        
        # Partition and save
        partition_file = self._get_partition_file(datetime.now())
        
        if partition_file.exists():
            existing = pd.read_parquet(partition_file)
            data = pd.concat([existing, data])
        
        data.to_parquet(partition_file)
        
        logger.info(f"Wrote {len(data)} records to {partition_file}")
        return len(data)
    
    def query_range(
        self,
        sensor_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query time series data for a time range.
        
        Args:
            sensor_id: Sensor identifier
            start_time: Start of time range
            end_time: End of time range
            aggregation: Optional aggregation (mean, sum, max, min)
            
        Returns:
            Time series data
        """
        results = []
        
        # Get relevant partitions
        for partition_file in self._get_partitions_for_range(start_time, end_time):
            if not partition_file.exists():
                continue
            
            df = pd.read_parquet(partition_file)
            
            # Filter by sensor and time
            df = df[
                (df['sensor_id'] == sensor_id) &
                (df.index >= start_time) &
                (df.index <= end_time)
            ]
            
            results.append(df)
        
        if not results:
            return pd.DataFrame()
        
        # Combine results
        combined = pd.concat(results)
        
        # Apply aggregation if requested
        if aggregation:
            combined = self._apply_aggregation(combined, aggregation)
        
        return combined
    
    def downsample(
        self,
        sensor_id: str,
        interval: str,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Downsample time series data.
        
        Args:
            sensor_id: Sensor identifier
            interval: Downsampling interval (e.g., '1H', '1D')
            aggregation: Aggregation method
            
        Returns:
            Downsampled data
        """
        # Get all data for sensor
        all_data = []
        
        for partition_file in self._get_partition_files():
            if not partition_file.exists():
                continue
            
            df = pd.read_parquet(partition_file)
            df = df[df['sensor_id'] == sensor_id]
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine and downsample
        combined = pd.concat(all_data)
        
        # Group by interval and aggregate
        resampled = combined.resample(interval).agg(aggregation)
        
        return resampled
    
    def get_statistics(
        self,
        sensor_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for sensor data.
        
        Args:
            sensor_id: Sensor identifier
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Statistics dictionary
        """
        # Query data
        if start_time and end_time:
            data = self.query_range(sensor_id, start_time, end_time)
        else:
            # Get all data for sensor
            criteria = {'sensor_id': sensor_id}
            entities = self.find_by(criteria)
            if not entities:
                return {}
            
            # Convert to DataFrame
            records = []
            for entity in entities:
                record = entity.values.copy()
                record['timestamp'] = entity.timestamp
                records.append(record)
            data = pd.DataFrame(records)
            data.set_index('timestamp', inplace=True)
        
        if data.empty:
            return {}
        
        # Calculate statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        stats = {
            'count': len(data),
            'start_time': data.index.min(),
            'end_time': data.index.max(),
            'duration': (data.index.max() - data.index.min()).total_seconds()
        }
        
        for col in numeric_cols:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            }
        
        return stats
    
    def cleanup_old_data(self) -> int:
        """
        Clean up data older than retention period.
        
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        for partition_file in self._get_partition_files():
            # Check partition date
            partition_date = self._get_partition_date(partition_file)
            
            if partition_date < cutoff_date:
                # Delete entire partition
                partition_file.unlink()
                logger.info(f"Deleted old partition: {partition_file}")
                deleted_count += 1
            else:
                # Clean records within partition
                df = pd.read_parquet(partition_file)
                original_len = len(df)
                df = df[df.index >= cutoff_date]
                
                if len(df) < original_len:
                    df.to_parquet(partition_file)
                    deleted_count += original_len - len(df)
        
        return deleted_count
    
    def _create_connection(self):
        """Create connection to data store."""
        return True
    
    def _close_connection(self):
        """Close connection and flush buffer."""
        self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        # Group by partition
        partitions = {}
        for entity in self._buffer:
            partition_file = self._get_partition_file(entity.timestamp)
            if partition_file not in partitions:
                partitions[partition_file] = []
            partitions[partition_file].append(entity)
        
        # Write each partition
        for partition_file, entities in partitions.items():
            # Convert to DataFrame
            records = []
            for entity in entities:
                record = entity.values.copy()
                record['sensor_id'] = entity.sensor_id
                record['timestamp'] = entity.timestamp
                records.append(record)
            
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            
            # Append to existing partition
            if partition_file.exists():
                existing = pd.read_parquet(partition_file)
                df = pd.concat([existing, df])
            
            df.to_parquet(partition_file)
        
        # Clear buffer
        self._buffer.clear()
    
    def _get_partition_file(self, timestamp: datetime) -> Path:
        """Get partition file for timestamp."""
        if self.partition_by == 'hour':
            partition = timestamp.strftime('%Y%m%d_%H')
        elif self.partition_by == 'day':
            partition = timestamp.strftime('%Y%m%d')
        elif self.partition_by == 'month':
            partition = timestamp.strftime('%Y%m')
        else:
            partition = 'default'
        
        return self.data_dir / f"partition_{partition}.parquet"
    
    def _get_partition_files(self) -> List[Path]:
        """Get all partition files."""
        return sorted(self.data_dir.glob("partition_*.parquet"))
    
    def _get_partitions_for_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Path]:
        """Get partition files for time range."""
        partitions = []
        
        current = start_time
        while current <= end_time:
            partitions.append(self._get_partition_file(current))
            
            # Move to next partition
            if self.partition_by == 'hour':
                current += timedelta(hours=1)
            elif self.partition_by == 'day':
                current += timedelta(days=1)
            elif self.partition_by == 'month':
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            else:
                break
        
        return list(set(partitions))
    
    def _get_partition_date(self, partition_file: Path) -> datetime:
        """Extract date from partition filename."""
        filename = partition_file.stem
        date_str = filename.replace('partition_', '')
        
        if self.partition_by == 'hour':
            return datetime.strptime(date_str, '%Y%m%d_%H')
        elif self.partition_by == 'day':
            return datetime.strptime(date_str, '%Y%m%d')
        elif self.partition_by == 'month':
            return datetime.strptime(date_str, '%Y%m')
        else:
            return datetime.now()
    
    def _row_to_entity(self, row: pd.Series) -> TimeSeriesEntity:
        """Convert DataFrame row to entity."""
        return TimeSeriesEntity(
            series_id=str(row.name),
            sensor_id=row.get('sensor_id', ''),
            timestamp=row.name if isinstance(row.name, datetime) else datetime.now(),
            values=row.to_dict(),
            metadata={}
        )
    
    def _apply_aggregation(self, df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
        """Apply aggregation to DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if aggregation == 'mean':
            return df[numeric_cols].mean()
        elif aggregation == 'sum':
            return df[numeric_cols].sum()
        elif aggregation == 'max':
            return df[numeric_cols].max()
        elif aggregation == 'min':
            return df[numeric_cols].min()
        elif aggregation == 'std':
            return df[numeric_cols].std()
        else:
            return df
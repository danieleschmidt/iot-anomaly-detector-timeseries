"""Multi-modal sensor fusion for comprehensive IoT anomaly detection."""

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from enum import Enum
import threading
import queue

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SensorType(Enum):
    """Types of sensors supported."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    LIGHT = "light"
    SOUND = "sound"
    GAS = "gas"
    FLOW = "flow"
    VOLTAGE = "voltage"
    CURRENT = "current"
    CUSTOM = "custom"


@dataclass
class SensorReading:
    """Individual sensor reading."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: float
    unit: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalData:
    """Multi-modal sensor data collection."""
    timestamp: float
    location_id: str
    readings: List[SensorReading]
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of multi-modal sensor fusion."""
    timestamp: float
    location_id: str
    overall_anomaly_score: float
    is_anomaly: bool
    confidence: float
    sensor_contributions: Dict[str, float]
    fusion_method: str
    processing_time_ms: float
    anomaly_type: Optional[str] = None
    severity_level: str = "low"


class SensorFusionConfig:
    """Configuration for sensor fusion."""
    
    def __init__(
        self,
        fusion_window_size: int = 10,
        correlation_threshold: float = 0.7,
        anomaly_threshold: float = 0.8,
        weight_decay_factor: float = 0.95,
        enable_temporal_correlation: bool = True,
        enable_spatial_correlation: bool = True,
        min_sensors_for_fusion: int = 2,
        sensor_weights: Optional[Dict[SensorType, float]] = None
    ):
        self.fusion_window_size = fusion_window_size
        self.correlation_threshold = correlation_threshold
        self.anomaly_threshold = anomaly_threshold
        self.weight_decay_factor = weight_decay_factor
        self.enable_temporal_correlation = enable_temporal_correlation
        self.enable_spatial_correlation = enable_spatial_correlation
        self.min_sensors_for_fusion = min_sensors_for_fusion
        
        # Default sensor weights
        self.sensor_weights = sensor_weights or {
            SensorType.TEMPERATURE: 1.0,
            SensorType.HUMIDITY: 0.8,
            SensorType.PRESSURE: 0.9,
            SensorType.VIBRATION: 1.2,
            SensorType.ACCELEROMETER: 1.1,
            SensorType.GYROSCOPE: 1.1,
            SensorType.VOLTAGE: 1.3,
            SensorType.CURRENT: 1.3,
        }


class MultiModalSensorFusion:
    """Advanced multi-modal sensor fusion for anomaly detection."""
    
    def __init__(self, config: Optional[SensorFusionConfig] = None):
        """Initialize multi-modal sensor fusion."""
        self.config = config or SensorFusionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.sensor_history: Dict[str, List[MultiModalData]] = {}
        self.sensor_baselines: Dict[str, Dict[SensorType, float]] = {}
        self.correlation_matrix: Dict[Tuple[SensorType, SensorType], float] = {}
        
        # Processing components
        self.processing_queue = queue.Queue()
        self.result_handlers: List[callable] = []
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics and monitoring
        self.fusion_stats = {
            "total_fusions": 0,
            "anomalies_detected": 0,
            "avg_processing_time_ms": 0.0,
            "sensor_reliability": {},
            "correlation_updates": 0
        }
        
        self._initialize_baseline_learning()
    
    def _initialize_baseline_learning(self) -> None:
        """Initialize baseline learning for normal operation patterns."""
        self.baseline_learning_samples = 100  # Number of samples to establish baseline
        self.baseline_update_interval = 50    # Update baseline every N samples
        self.baseline_sample_counts: Dict[str, int] = {}
    
    def start_processing(self) -> None:
        """Start background processing of sensor fusion."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        self.logger.info("Started multi-modal sensor fusion processing")
    
    def stop_processing(self) -> None:
        """Stop background processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.logger.info("Stopped multi-modal sensor fusion processing")
    
    def add_sensor_data(self, multi_modal_data: MultiModalData) -> None:
        """Add multi-modal sensor data for fusion processing."""
        try:
            self.processing_queue.put_nowait(multi_modal_data)
        except queue.Full:
            self.logger.warning("Processing queue full, dropping sensor data")
    
    def add_result_handler(self, handler: callable) -> None:
        """Add handler for fusion results."""
        self.result_handlers.append(handler)
    
    def _processing_loop(self) -> None:
        """Main processing loop for sensor fusion."""
        while self.is_processing:
            try:
                try:
                    data = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the multi-modal data
                result = self._process_multi_modal_data(data)
                if result:
                    self._handle_fusion_result(result)
                    
            except Exception as e:
                self.logger.error(f"Error in fusion processing loop: {e}")
                time.sleep(0.1)
    
    def _process_multi_modal_data(self, data: MultiModalData) -> Optional[FusionResult]:
        """Process multi-modal sensor data and perform fusion."""
        start_time = time.time()
        
        try:
            # Update sensor history
            location_id = data.location_id
            if location_id not in self.sensor_history:
                self.sensor_history[location_id] = []
            
            self.sensor_history[location_id].append(data)
            
            # Maintain history window
            max_history = self.config.fusion_window_size * 2
            if len(self.sensor_history[location_id]) > max_history:
                self.sensor_history[location_id] = self.sensor_history[location_id][-max_history:]
            
            # Update baselines if in learning phase
            self._update_sensor_baselines(location_id, data)
            
            # Perform sensor fusion if we have enough data
            if len(self.sensor_history[location_id]) >= self.config.min_sensors_for_fusion:
                fusion_result = self._perform_sensor_fusion(location_id, data)
                
                processing_time = (time.time() - start_time) * 1000
                fusion_result.processing_time_ms = processing_time
                
                return fusion_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing multi-modal data: {e}")
            return None
    
    def _update_sensor_baselines(self, location_id: str, data: MultiModalData) -> None:
        """Update baseline values for sensor readings."""
        if location_id not in self.sensor_baselines:
            self.sensor_baselines[location_id] = {}
        
        if location_id not in self.baseline_sample_counts:
            self.baseline_sample_counts[location_id] = 0
        
        # Only update baselines during learning phase
        sample_count = self.baseline_sample_counts[location_id]
        if sample_count < self.baseline_learning_samples:
            for reading in data.readings:
                sensor_type = reading.sensor_type
                
                if sensor_type not in self.sensor_baselines[location_id]:
                    self.sensor_baselines[location_id][sensor_type] = reading.value
                else:
                    # Running average
                    current_baseline = self.sensor_baselines[location_id][sensor_type]
                    alpha = 0.1  # Learning rate
                    self.sensor_baselines[location_id][sensor_type] = (
                        (1 - alpha) * current_baseline + alpha * reading.value
                    )
            
            self.baseline_sample_counts[location_id] += 1
        
        # Periodic baseline updates after learning phase
        elif sample_count % self.baseline_update_interval == 0:
            self._gradual_baseline_update(location_id, data)
    
    def _gradual_baseline_update(self, location_id: str, data: MultiModalData) -> None:
        """Gradually update baselines to adapt to environmental changes."""
        alpha = 0.01  # Very slow adaptation
        
        for reading in data.readings:
            sensor_type = reading.sensor_type
            if sensor_type in self.sensor_baselines[location_id]:
                current_baseline = self.sensor_baselines[location_id][sensor_type]
                # Only update if reading is not anomalous
                deviation = abs(reading.value - current_baseline) / (current_baseline + 1e-8)
                if deviation < 0.5:  # Not too far from baseline
                    self.sensor_baselines[location_id][sensor_type] = (
                        (1 - alpha) * current_baseline + alpha * reading.value
                    )
    
    def _perform_sensor_fusion(self, location_id: str, current_data: MultiModalData) -> FusionResult:
        """Perform multi-modal sensor fusion analysis."""
        # Get recent sensor history
        history = self.sensor_history[location_id][-self.config.fusion_window_size:]
        
        # Calculate individual sensor anomaly scores
        sensor_scores = self._calculate_individual_sensor_scores(location_id, current_data)
        
        # Calculate temporal correlations
        temporal_scores = {}
        if self.config.enable_temporal_correlation:
            temporal_scores = self._calculate_temporal_correlations(history, current_data)
        
        # Calculate spatial correlations (cross-sensor)
        spatial_scores = {}
        if self.config.enable_spatial_correlation:
            spatial_scores = self._calculate_spatial_correlations(current_data)
        
        # Fusion using weighted combination
        overall_score = self._weighted_fusion(sensor_scores, temporal_scores, spatial_scores)
        
        # Determine anomaly classification
        is_anomaly = overall_score > self.config.anomaly_threshold
        confidence = min(overall_score / self.config.anomaly_threshold, 2.0)
        
        # Determine anomaly type and severity
        anomaly_type, severity_level = self._classify_anomaly(sensor_scores, overall_score)
        
        result = FusionResult(
            timestamp=current_data.timestamp,
            location_id=location_id,
            overall_anomaly_score=overall_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            sensor_contributions=sensor_scores,
            fusion_method="weighted_multi_modal",
            processing_time_ms=0.0,  # Will be set by caller
            anomaly_type=anomaly_type,
            severity_level=severity_level
        )
        
        return result
    
    def _calculate_individual_sensor_scores(
        self, 
        location_id: str, 
        data: MultiModalData
    ) -> Dict[str, float]:
        """Calculate anomaly scores for individual sensors."""
        scores = {}
        baselines = self.sensor_baselines.get(location_id, {})
        
        for reading in data.readings:
            sensor_key = f"{reading.sensor_id}_{reading.sensor_type.value}"
            
            if reading.sensor_type in baselines:
                baseline = baselines[reading.sensor_type]
                
                # Calculate normalized deviation
                deviation = abs(reading.value - baseline)
                normalized_score = deviation / (baseline + 1e-8)
                
                # Apply sensor-specific weighting
                weight = self.config.sensor_weights.get(reading.sensor_type, 1.0)
                weighted_score = normalized_score * weight
                
                # Apply confidence scaling
                final_score = weighted_score * reading.confidence
                scores[sensor_key] = min(final_score, 2.0)  # Cap at 2.0
            else:
                # No baseline available, use default moderate score
                scores[sensor_key] = 0.3
        
        return scores
    
    def _calculate_temporal_correlations(
        self, 
        history: List[MultiModalData], 
        current_data: MultiModalData
    ) -> Dict[str, float]:
        """Calculate temporal correlation anomalies."""
        if not NUMPY_AVAILABLE or len(history) < 3:
            return {}
        
        temporal_scores = {}
        
        # Group readings by sensor type
        sensor_time_series = {}
        for data_point in history:
            for reading in data_point.readings:
                key = f"{reading.sensor_id}_{reading.sensor_type.value}"
                if key not in sensor_time_series:
                    sensor_time_series[key] = []
                sensor_time_series[key].append(reading.value)
        
        # Calculate temporal anomalies
        for sensor_key, values in sensor_time_series.items():
            if len(values) >= 3:
                values_array = np.array(values)
                
                # Calculate trend and seasonality
                trend_score = self._calculate_trend_anomaly(values_array)
                volatility_score = self._calculate_volatility_anomaly(values_array)
                
                temporal_scores[sensor_key] = max(trend_score, volatility_score)
        
        return temporal_scores
    
    def _calculate_trend_anomaly(self, values: np.ndarray) -> float:
        """Calculate trend-based anomaly score."""
        if len(values) < 3:
            return 0.0
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate recent vs historical trend
        if len(values) >= 6:
            recent_slope = np.polyfit(x[-3:], values[-3:], 1)[0]
            historical_slope = np.polyfit(x[:-3], values[:-3], 1)[0]
            
            trend_change = abs(recent_slope - historical_slope)
            return min(trend_change / (abs(historical_slope) + 1e-8), 1.0)
        
        return 0.0
    
    def _calculate_volatility_anomaly(self, values: np.ndarray) -> float:
        """Calculate volatility-based anomaly score."""
        if len(values) < 3:
            return 0.0
        
        # Calculate rolling standard deviation
        window_size = min(3, len(values) // 2)
        if window_size < 2:
            return 0.0
        
        recent_std = np.std(values[-window_size:])
        historical_std = np.std(values[:-window_size]) if len(values) > window_size else recent_std
        
        if historical_std > 1e-8:
            volatility_ratio = recent_std / historical_std
            # Anomaly if volatility changed significantly
            return min(abs(volatility_ratio - 1.0), 1.0)
        
        return 0.0
    
    def _calculate_spatial_correlations(self, data: MultiModalData) -> Dict[str, float]:
        """Calculate spatial (cross-sensor) correlation anomalies."""
        if len(data.readings) < 2:
            return {}
        
        spatial_scores = {}
        
        # Check for expected correlations between sensor types
        sensor_pairs = [
            (SensorType.TEMPERATURE, SensorType.HUMIDITY),
            (SensorType.VOLTAGE, SensorType.CURRENT),
            (SensorType.ACCELEROMETER, SensorType.VIBRATION),
            (SensorType.PRESSURE, SensorType.TEMPERATURE)
        ]
        
        readings_by_type = {
            reading.sensor_type: reading.value 
            for reading in data.readings
        }
        
        for sensor_type1, sensor_type2 in sensor_pairs:
            if sensor_type1 in readings_by_type and sensor_type2 in readings_by_type:
                correlation_score = self._check_sensor_correlation(
                    sensor_type1, sensor_type2,
                    readings_by_type[sensor_type1],
                    readings_by_type[sensor_type2]
                )
                
                key = f"{sensor_type1.value}_{sensor_type2.value}_correlation"
                spatial_scores[key] = correlation_score
        
        return spatial_scores
    
    def _check_sensor_correlation(
        self, 
        sensor1: SensorType, 
        sensor2: SensorType,
        value1: float, 
        value2: float
    ) -> float:
        """Check correlation between two sensor readings."""
        # Simple correlation rules (can be enhanced with ML)
        correlation_rules = {
            (SensorType.TEMPERATURE, SensorType.HUMIDITY): self._temp_humidity_correlation,
            (SensorType.VOLTAGE, SensorType.CURRENT): self._voltage_current_correlation,
            (SensorType.ACCELEROMETER, SensorType.VIBRATION): self._motion_correlation
        }
        
        key = (sensor1, sensor2)
        reverse_key = (sensor2, sensor1)
        
        if key in correlation_rules:
            return correlation_rules[key](value1, value2)
        elif reverse_key in correlation_rules:
            return correlation_rules[reverse_key](value2, value1)
        
        return 0.0  # No correlation rule defined
    
    def _temp_humidity_correlation(self, temp: float, humidity: float) -> float:
        """Check temperature-humidity correlation (inverse relationship expected)."""
        # Simple rule: high temperature usually means lower relative humidity
        expected_humidity = max(20, 80 - temp * 1.5)  # Simple linear model
        deviation = abs(humidity - expected_humidity) / expected_humidity
        return min(deviation, 1.0)
    
    def _voltage_current_correlation(self, voltage: float, current: float) -> float:
        """Check voltage-current correlation (should follow Ohm's law patterns)."""
        # For most loads, expect some proportional relationship
        if voltage > 1e-8:
            expected_current_range = (voltage * 0.1, voltage * 2.0)  # Wide range
            if not (expected_current_range[0] <= current <= expected_current_range[1]):
                return 0.8  # High anomaly score for unexpected V-I relationship
        return 0.0
    
    def _motion_correlation(self, acceleration: float, vibration: float) -> float:
        """Check acceleration-vibration correlation."""
        # Expect some correlation between acceleration and vibration
        expected_vibration = abs(acceleration) * 0.5  # Simple model
        deviation = abs(vibration - expected_vibration) / (expected_vibration + 1e-8)
        return min(deviation, 1.0)
    
    def _weighted_fusion(
        self, 
        sensor_scores: Dict[str, float],
        temporal_scores: Dict[str, float],
        spatial_scores: Dict[str, float]
    ) -> float:
        """Perform weighted fusion of all anomaly scores."""
        all_scores = []
        
        # Add sensor scores with high weight
        all_scores.extend([score * 1.0 for score in sensor_scores.values()])
        
        # Add temporal scores with medium weight
        all_scores.extend([score * 0.7 for score in temporal_scores.values()])
        
        # Add spatial scores with medium weight
        all_scores.extend([score * 0.8 for score in spatial_scores.values()])
        
        if not all_scores:
            return 0.0
        
        # Use combination of max and average for robustness
        max_score = max(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        # Weighted combination favoring max score but considering average
        fusion_score = 0.7 * max_score + 0.3 * avg_score
        
        return min(fusion_score, 2.0)  # Cap at 2.0
    
    def _classify_anomaly(self, sensor_scores: Dict[str, float], overall_score: float) -> Tuple[Optional[str], str]:
        """Classify type and severity of detected anomaly."""
        if overall_score <= self.config.anomaly_threshold:
            return None, "normal"
        
        # Determine dominant contributing sensor
        if sensor_scores:
            max_sensor = max(sensor_scores.keys(), key=lambda k: sensor_scores[k])
            sensor_type = max_sensor.split('_')[-1]
            
            # Classify severity
            if overall_score > 1.5:
                severity = "high"
            elif overall_score > 1.0:
                severity = "medium"
            else:
                severity = "low"
            
            return f"{sensor_type}_anomaly", severity
        
        return "unknown_anomaly", "low"
    
    def _handle_fusion_result(self, result: FusionResult) -> None:
        """Handle fusion result and update statistics."""
        # Update statistics
        self.fusion_stats["total_fusions"] += 1
        if result.is_anomaly:
            self.fusion_stats["anomalies_detected"] += 1
        
        # Update average processing time
        current_avg = self.fusion_stats["avg_processing_time_ms"]
        total_fusions = self.fusion_stats["total_fusions"]
        new_avg = ((current_avg * (total_fusions - 1)) + result.processing_time_ms) / total_fusions
        self.fusion_stats["avg_processing_time_ms"] = new_avg
        
        # Call result handlers
        for handler in self.result_handlers:
            try:
                handler(result)
            except Exception as e:
                self.logger.error(f"Error in fusion result handler: {e}")
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion processing statistics."""
        return {
            **self.fusion_stats,
            "active_locations": len(self.sensor_history),
            "baseline_learning_progress": {
                loc: min(count / self.baseline_learning_samples, 1.0)
                for loc, count in self.baseline_sample_counts.items()
            },
            "queue_size": self.processing_queue.qsize()
        }
    
    def export_baselines(self, output_path: str) -> None:
        """Export learned baselines to file."""
        export_data = {
            "baselines": {
                loc: {sensor_type.value: baseline for sensor_type, baseline in baselines.items()}
                for loc, baselines in self.sensor_baselines.items()
            },
            "sample_counts": self.baseline_sample_counts,
            "export_timestamp": time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported baselines to {output_path}")
    
    def import_baselines(self, input_path: str) -> None:
        """Import baselines from file."""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Convert string keys back to SensorType enums
            for loc, baselines in import_data["baselines"].items():
                self.sensor_baselines[loc] = {
                    SensorType(sensor_type): baseline 
                    for sensor_type, baseline in baselines.items()
                }
            
            self.baseline_sample_counts.update(import_data.get("sample_counts", {}))
            
            self.logger.info(f"Imported baselines from {input_path}")
            
        except Exception as e:
            self.logger.error(f"Error importing baselines: {e}")


# Factory function
def create_sensor_fusion(
    fusion_window_size: int = 10,
    anomaly_threshold: float = 0.8,
    **kwargs
) -> MultiModalSensorFusion:
    """Create multi-modal sensor fusion instance."""
    config = SensorFusionConfig(
        fusion_window_size=fusion_window_size,
        anomaly_threshold=anomaly_threshold,
        **kwargs
    )
    return MultiModalSensorFusion(config)


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Multi-Modal Sensor Fusion")
    parser.add_argument("--fusion-window", type=int, default=10, help="Fusion window size")
    parser.add_argument("--anomaly-threshold", type=float, default=0.8, help="Anomaly threshold")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--export-baselines", help="Export baselines to file")
    
    args = parser.parse_args()
    
    # Create fusion system
    fusion = create_sensor_fusion(
        fusion_window_size=args.fusion_window,
        anomaly_threshold=args.anomaly_threshold
    )
    
    # Add result handler
    def print_anomaly(result: FusionResult):
        if result.is_anomaly:
            print(f"ANOMALY at {result.location_id}: score={result.overall_anomaly_score:.2f}, "
                  f"type={result.anomaly_type}, severity={result.severity_level}")
    
    fusion.add_result_handler(print_anomaly)
    fusion.start_processing()
    
    print(f"Starting multi-modal sensor fusion test for {args.duration} seconds...")
    
    # Simulate multi-modal sensor data
    start_time = time.time()
    sample_count = 0
    
    while time.time() - start_time < args.duration:
        # Create realistic multi-modal data
        timestamp = time.time()
        
        # Simulate correlated sensor readings
        base_temp = 25 + 5 * np.sin(sample_count * 0.1) if NUMPY_AVAILABLE else 25 + random.gauss(0, 2)
        
        readings = [
            SensorReading("temp_01", SensorType.TEMPERATURE, timestamp, base_temp, "C"),
            SensorReading("hum_01", SensorType.HUMIDITY, timestamp, 
                         max(20, 80 - base_temp * 1.2 + random.gauss(0, 5)), "%"),
            SensorReading("volt_01", SensorType.VOLTAGE, timestamp, 
                         12.0 + random.gauss(0, 0.5), "V"),
            SensorReading("curr_01", SensorType.CURRENT, timestamp, 
                         2.0 + random.gauss(0, 0.2), "A")
        ]
        
        # Inject occasional anomalies
        if random.random() < 0.1:  # 10% anomaly rate
            anomaly_sensor = random.choice(readings)
            anomaly_sensor.value *= 1.5 + random.random()  # Significant deviation
        
        multi_modal_data = MultiModalData(
            timestamp=timestamp,
            location_id="facility_001",
            readings=readings,
            environmental_context={"weather": "normal"},
            system_context={"load": "normal"}
        )
        
        fusion.add_sensor_data(multi_modal_data)
        
        sample_count += 1
        time.sleep(0.1)  # 10 Hz sampling
    
    fusion.stop_processing()
    
    # Print statistics
    stats = fusion.get_fusion_statistics()
    print(f"\nFusion Statistics:")
    print(f"Total fusions: {stats['total_fusions']}")
    print(f"Anomalies detected: {stats['anomalies_detected']}")
    print(f"Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
    
    # Export baselines if requested
    if args.export_baselines:
        fusion.export_baselines(args.export_baselines)
        print(f"Baselines exported to {args.export_baselines}")
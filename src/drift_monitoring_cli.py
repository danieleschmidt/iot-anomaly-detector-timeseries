#!/usr/bin/env python3
"""
Data Drift Monitoring CLI for IoT Anomaly Detection System

This CLI provides real-time data drift monitoring, alerting, and analysis
capabilities for maintaining model accuracy in production environments.
"""

import argparse
import json
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .data_drift_detector import (
    DataDriftDetector, 
    DriftDetectionConfig, 
    create_drift_detector_from_training_data
)
from .logging_config import get_logger

logger = get_logger(__name__)


def format_drift_result(result, detailed: bool = False) -> str:
    """Format drift detection result for display."""
    output = []
    output.append("=" * 70)
    output.append(f"DRIFT DETECTION RESULT - {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 70)
    
    # Overall status
    status = "🚨 DRIFT DETECTED" if result.drift_detected else "✅ NO DRIFT DETECTED"
    output.append(f"Status: {status}")
    output.append("")
    
    # Key metrics
    output.append("KEY METRICS:")
    output.append(f"  Overall Drift:           {'Yes' if result.drift_detected else 'No'}")
    output.append(f"  Features Analyzed:       {len(result.feature_drifts)}")
    output.append(f"  Features with Drift:     {sum(result.feature_drifts.values())}")
    output.append(f"  Drift Rate:              {result.summary.get('drift_rate', 0):.1%}")
    output.append("")
    
    # Statistical scores
    output.append("STATISTICAL SCORES:")
    output.append(f"  Max PSI Score:           {result.psi_score:.4f}")
    output.append(f"  Min KS p-value:          {result.ks_p_value:.6f}")
    output.append(f"  Max Wasserstein Dist:    {result.wasserstein_distance:.4f}")
    output.append(f"  Avg KS Statistic:        {result.ks_statistic:.4f}")
    
    if detailed and result.drift_detected:
        output.append("")
        output.append("FEATURE-LEVEL ANALYSIS:")
        output.append("-" * 50)
        
        drifted_features = [(f, scores) for f, scores in result.drift_scores.items() 
                           if result.feature_drifts.get(f, False)]
        
        if drifted_features:
            for feature, scores in drifted_features:
                output.append(f"\n📊 {feature}:")
                output.append(f"    PSI Score:        {scores.get('psi', 0):.4f}")
                output.append(f"    KS p-value:       {scores.get('ks_pvalue', 1):.6f}")
                output.append(f"    Wasserstein:      {scores.get('wasserstein', 0):.4f}")
                
                # Interpretation
                psi = scores.get('psi', 0)
                if psi > 0.25:
                    interpretation = "Significant change"
                elif psi > 0.1:
                    interpretation = "Moderate change"
                else:
                    interpretation = "Slight change"
                output.append(f"    Interpretation:   {interpretation}")
        
        # Most problematic feature
        if result.summary.get('max_psi_feature'):
            output.append(f"\n🎯 Most Drifted Feature: {result.summary['max_psi_feature']}")
    
    output.append("\n" + "=" * 70)
    return "\n".join(output)


def monitor_drift_continuously(
    detector: DataDriftDetector,
    data_source: str,
    interval_minutes: int = 60,
    duration_hours: int = 24,
    output_dir: Optional[str] = None
) -> None:
    """
    Monitor data drift continuously from a data source.
    
    Parameters
    ----------
    detector : DataDriftDetector
        Configured drift detector
    data_source : str
        Path to data file or directory to monitor
    interval_minutes : int
        Check interval in minutes
    duration_hours : int
        Total monitoring duration in hours
    output_dir : str, optional
        Directory to save monitoring results
    """
    print(f"🔍 Starting continuous drift monitoring...")
    print(f"📊 Data source: {data_source}")
    print(f"⏱️  Check interval: {interval_minutes} minutes")
    print(f"⏰ Duration: {duration_hours} hours")
    print("Press Ctrl+C to stop monitoring early.\n")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    check_count = 0
    drift_alerts = 0
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        logger.info(f"Monitoring results will be saved to {output_path}")
    
    try:
        while datetime.now() < end_time:
            check_count += 1
            current_time = datetime.now()
            
            print(f"\n🕐 Check #{check_count} at {current_time.strftime('%H:%M:%S')}")
            
            try:
                # Load new data (simulate reading from data source)
                if Path(data_source).exists():
                    new_data = pd.read_csv(data_source)
                    # In a real scenario, you might read only recent data
                    # For demo, we'll sample the data differently each time
                    sample_size = min(1000, len(new_data))
                    new_data = new_data.sample(n=sample_size, random_state=check_count)
                    
                    # Detect drift
                    result = detector.detect_drift(new_data)
                    
                    # Display result
                    if result.drift_detected:
                        drift_alerts += 1
                        print("🚨 DRIFT ALERT!")
                        print(format_drift_result(result, detailed=True))
                        
                        # Save alert if output directory specified
                        if output_dir:
                            alert_file = output_path / f"drift_alert_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                            with open(alert_file, 'w') as f:
                                json.dump(result.to_dict(), f, indent=2, default=str)
                    else:
                        print("✅ No drift detected")
                        print(f"   Max PSI: {result.psi_score:.4f}, Min KS p-value: {result.ks_p_value:.6f}")
                    
                    # Show summary every 5 checks
                    if check_count % 5 == 0:
                        summary = detector.get_drift_summary()
                        print(f"\n📈 MONITORING SUMMARY (last {summary['period_days']} days):")
                        print(f"   Total checks: {check_count}")
                        print(f"   Drift alerts: {drift_alerts}")
                        print(f"   Alert rate: {drift_alerts/check_count:.1%}")
                        print(f"   Most drifted feature: {summary.get('most_drifted_feature', 'None')}")
                
                except Exception as e:
                    print(f"❌ Error during drift check: {e}")
                    logger.error(f"Drift monitoring error: {e}")
            
            # Wait for next check
            if datetime.now() < end_time:
                sleep_seconds = interval_minutes * 60
                print(f"💤 Sleeping for {interval_minutes} minutes...")
                time.sleep(sleep_seconds)
    
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    # Final summary
    total_time = datetime.now() - start_time
    print(f"\n📊 FINAL MONITORING SUMMARY:")
    print(f"   Duration: {total_time}")
    print(f"   Total checks: {check_count}")
    print(f"   Drift alerts: {drift_alerts}")
    print(f"   Alert rate: {drift_alerts/check_count:.1%}" if check_count > 0 else "   Alert rate: N/A")
    
    # Export final results
    if output_dir:
        final_export = output_path / f"drift_monitoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        detector.export_drift_history(str(final_export))
        print(f"📁 Results exported to {final_export}")


def analyze_drift_trends(
    history_file: str,
    days: int = 30,
    export_path: Optional[str] = None
) -> None:
    """
    Analyze drift trends from historical data.
    
    Parameters
    ----------
    history_file : str
        Path to drift history JSON file
    days : int
        Number of days to analyze
    export_path : str, optional
        Path to export analysis results
    """
    print(f"📈 Analyzing drift trends from {history_file}")
    
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        drift_history = history_data.get('drift_history', [])
        if not drift_history:
            print("❌ No drift history found in file")
            return
        
        print(f"📊 Loaded {len(drift_history)} drift detection records")
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = []
        
        for record in drift_history:
            record_time = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
            if record_time >= cutoff_date:
                recent_history.append(record)
        
        if not recent_history:
            print(f"❌ No records found in the last {days} days")
            return
        
        print(f"🔍 Analyzing {len(recent_history)} records from last {days} days")
        
        # Trend analysis
        drift_rates = []
        timestamps = []
        
        for record in recent_history:
            drift_rates.append(record['summary'].get('drift_rate', 0))
            timestamps.append(datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')))
        
        # Calculate trends
        avg_drift_rate = np.mean(drift_rates)
        trend_slope = np.polyfit(range(len(drift_rates)), drift_rates, 1)[0] if len(drift_rates) > 1 else 0
        
        print("\n" + "=" * 60)
        print("DRIFT TREND ANALYSIS")
        print("=" * 60)
        print(f"Analysis Period:     {days} days")
        print(f"Total Records:       {len(recent_history)}")
        print(f"Average Drift Rate:  {avg_drift_rate:.2%}")
        print(f"Trend Direction:     {'📈 Increasing' if trend_slope > 0.01 else '📉 Decreasing' if trend_slope < -0.01 else '➡️ Stable'}")
        print(f"Trend Slope:         {trend_slope:.6f}")
        
        # Feature analysis
        feature_drift_counts = {}
        total_checks = len(recent_history)
        
        for record in recent_history:
            for feature, drifted in record.get('feature_drifts', {}).items():
                if drifted:
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
        
        if feature_drift_counts:
            print(f"\n📊 FEATURE DRIFT FREQUENCY:")
            sorted_features = sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)
            
            for feature, count in sorted_features[:10]:  # Top 10
                rate = count / total_checks
                print(f"   {feature:20} {count:3d} times ({rate:.1%})")
        
        # Time-based patterns
        hour_counts = {}
        for timestamp in timestamps:
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])
            print(f"\n⏰ TEMPORAL PATTERNS:")
            print(f"   Peak drift hour:     {peak_hour[0]:02d}:00 ({peak_hour[1]} detections)")
        
        # Severity analysis
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for record in recent_history:
            psi_score = record.get('psi_score', 0)
            if psi_score > 0.5:
                severity_counts['critical'] += 1
            elif psi_score > 0.25:
                severity_counts['high'] += 1
            elif record.get('drift_detected', False):
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        print(f"\n🚨 SEVERITY DISTRIBUTION:")
        for severity, count in severity_counts.items():
            rate = count / total_checks
            print(f"   {severity.capitalize():8} {count:3d} ({rate:.1%})")
        
        # Export analysis if requested
        if export_path:
            analysis_results = {
                'analysis_date': datetime.now().isoformat(),
                'period_days': days,
                'total_records': len(recent_history),
                'average_drift_rate': avg_drift_rate,
                'trend_slope': trend_slope,
                'feature_drift_counts': feature_drift_counts,
                'severity_distribution': severity_counts,
                'temporal_patterns': hour_counts
            }
            
            with open(export_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            print(f"\n📁 Analysis exported to {export_path}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing drift trends: {e}")
        logger.error(f"Drift trend analysis failed: {e}")


def setup_drift_monitoring(
    training_data_path: str,
    config_path: Optional[str] = None
) -> DataDriftDetector:
    """
    Set up drift detector from training data and configuration.
    
    Parameters
    ----------
    training_data_path : str
        Path to training data CSV
    config_path : str, optional
        Path to configuration JSON file
        
    Returns
    -------
    DataDriftDetector
        Configured drift detector
    """
    print(f"⚙️  Setting up drift detector...")
    
    # Load configuration if provided
    config = DriftDetectionConfig()
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            config = DriftDetectionConfig(**config_data)
            print(f"📋 Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to load config from {config_path}: {e}")
            print("Using default configuration")
    
    # Create detector
    try:
        detector = create_drift_detector_from_training_data(training_data_path, config)
        print(f"✅ Drift detector ready!")
        print(f"   Training data: {training_data_path}")
        print(f"   Reference samples: {len(detector.reference_data)}")
        print(f"   Features: {list(detector.reference_data.columns)}")
        
        return detector
        
    except Exception as e:
        print(f"❌ Failed to set up drift detector: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Drift Monitoring CLI for IoT Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start continuous drift monitoring')
    monitor_parser.add_argument(
        '--training-data', '-t',
        required=True,
        help='Path to training data CSV file'
    )
    monitor_parser.add_argument(
        '--data-source', '-s',
        required=True,
        help='Path to data source to monitor'
    )
    monitor_parser.add_argument(
        '--config', '-c',
        help='Path to configuration JSON file'
    )
    monitor_parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Check interval in minutes (default: 60)'
    )
    monitor_parser.add_argument(
        '--duration', '-d',
        type=int,
        default=24,
        help='Monitoring duration in hours (default: 24)'
    )
    monitor_parser.add_argument(
        '--output-dir', '-o',
        help='Directory to save monitoring results'
    )
    
    # Check command (single check)
    check_parser = subparsers.add_parser('check', help='Perform single drift check')
    check_parser.add_argument(
        '--training-data', '-t',
        required=True,
        help='Path to training data CSV file'
    )
    check_parser.add_argument(
        '--new-data', '-n',
        required=True,
        help='Path to new data CSV file to check for drift'
    )
    check_parser.add_argument(
        '--config', '-c',
        help='Path to configuration JSON file'
    )
    check_parser.add_argument(
        '--detailed', '-v',
        action='store_true',
        help='Show detailed drift analysis'
    )
    check_parser.add_argument(
        '--export', '-e',
        help='Export results to JSON file'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze drift trends from history')
    analyze_parser.add_argument(
        '--history-file', '-f',
        required=True,
        help='Path to drift history JSON file'
    )
    analyze_parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    analyze_parser.add_argument(
        '--export', '-e',
        help='Export analysis to JSON file'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'monitor':
            detector = setup_drift_monitoring(args.training_data, args.config)
            monitor_drift_continuously(
                detector=detector,
                data_source=args.data_source,
                interval_minutes=args.interval,
                duration_hours=args.duration,
                output_dir=args.output_dir
            )
        
        elif args.command == 'check':
            detector = setup_drift_monitoring(args.training_data, args.config)
            
            # Load and check new data
            new_data = pd.read_csv(args.new_data)
            print(f"📊 Checking {len(new_data)} samples for drift...")
            
            result = detector.detect_drift(new_data)
            print(format_drift_result(result, detailed=args.detailed))
            
            if args.export:
                with open(args.export, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                print(f"📁 Results exported to {args.export}")
        
        elif args.command == 'analyze':
            analyze_drift_trends(
                history_file=args.history_file,
                days=args.days,
                export_path=args.export
            )
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"CLI command failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Command-line interface for performance monitoring and metrics analysis."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from .logging_config import get_performance_metrics, get_logger


class PerformanceMonitorCLI:
    """Command-line interface for performance monitoring."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_performance_metrics()
    
    def show_live_metrics(self, refresh_interval: float = 5.0, max_iterations: int = None) -> None:
        """Display live performance metrics.
        
        Parameters
        ----------
        refresh_interval : float
            Seconds between metric updates
        max_iterations : int, optional
            Maximum number of iterations (None for infinite)
        """
        print("🔄 Live Performance Monitoring")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                # Get current metrics
                stats = self.metrics.get_summary_stats()
                
                # Display header
                print("=" * 80)
                print(f"📊 Performance Metrics - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                
                # System uptime
                uptime_hours = stats.get('uptime_seconds', 0) / 3600
                print(f"⏱️  System Uptime: {uptime_hours:.2f} hours")
                
                # Memory metrics
                if 'memory' in stats:
                    memory = stats['memory']
                    print(f"💾 Memory Usage:")
                    print(f"   RSS: {memory['current_rss_mb']:.1f} MB")
                    print(f"   VMS: {memory['current_vms_mb']:.1f} MB")
                    print(f"   Percent: {memory['current_percent']:.1f}%")
                
                # GPU metrics
                if 'gpu' in stats:
                    gpu = stats['gpu']
                    print(f"🎮 GPU Status:")
                    print(f"   Utilization: {gpu['utilization_percent']:.1f}%")
                    print(f"   Memory: {gpu['memory_used_mb']:.0f}/{gpu.get('memory_total_mb', 0):.0f} MB ({gpu['memory_percent']:.1f}%)")
                    print(f"   Temperature: {gpu['temperature_c']:.1f}°C")
                
                # Timing statistics
                if 'timing' in stats:
                    timing = stats['timing']
                    print(f"⏱️  Operation Timing:")
                    print(f"   Count: {timing['count']}")
                    print(f"   Avg: {timing['avg']:.3f}s")
                    print(f"   Min: {timing['min']:.3f}s")
                    print(f"   Max: {timing['max']:.3f}s")
                    print(f"   Total: {timing['total']:.2f}s")
                
                # Counters
                if 'counters' in stats and stats['counters']:
                    print(f"🔢 Counters:")
                    for name, count in sorted(stats['counters'].items()):
                        if count > 0:
                            print(f"   {name}: {count}")
                
                print("\n⏱️  Next update in {:.1f}s...".format(refresh_interval))
                
                # Wait for next iteration
                time.sleep(refresh_interval)
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
    
    def show_summary(self, operation: str = None, last_n: int = None) -> None:
        """Show performance summary.
        
        Parameters
        ----------
        operation : str, optional
            Filter by operation name
        last_n : int, optional
            Show only last N entries
        """
        stats = self.metrics.get_summary_stats(operation=operation, last_n=last_n)
        
        print("📊 Performance Summary")
        print("=" * 50)
        
        if operation:
            print(f"🔍 Filtered by operation: {operation}")
        if last_n:
            print(f"📉 Showing last {last_n} entries")
        
        print(f"⏱️  System Uptime: {stats.get('uptime_seconds', 0) / 3600:.2f} hours")
        
        # Timing stats
        if 'timing' in stats:
            timing = stats['timing']
            print(f"\n⏱️  Timing Statistics:")
            print(f"   Operations: {timing['count']}")
            print(f"   Average: {timing['avg']:.3f}s")
            print(f"   Min: {timing['min']:.3f}s")
            print(f"   Max: {timing['max']:.3f}s")
            print(f"   Total: {timing['total']:.2f}s")
            
            if timing['count'] > 0:
                rate = timing['count'] / stats.get('uptime_seconds', 1)
                print(f"   Rate: {rate:.2f} ops/sec")
        
        # Memory stats
        if 'memory' in stats:
            memory = stats['memory']
            print(f"\n💾 Current Memory Usage:")
            print(f"   RSS: {memory['current_rss_mb']:.1f} MB")
            print(f"   VMS: {memory['current_vms_mb']:.1f} MB")
            print(f"   Percent: {memory['current_percent']:.1f}%")
        
        # GPU stats
        if 'gpu' in stats:
            gpu = stats['gpu']
            print(f"\n🎮 Current GPU Status:")
            print(f"   Utilization: {gpu['utilization_percent']:.1f}%")
            print(f"   Memory Used: {gpu['memory_used_mb']:.0f} MB")
            print(f"   Memory Percent: {gpu['memory_percent']:.1f}%")
            print(f"   Temperature: {gpu['temperature_c']:.1f}°C")
        
        # Counters
        if 'counters' in stats and stats['counters']:
            print(f"\n🔢 Counters:")
            for name, count in sorted(stats['counters'].items()):
                if count > 0:
                    print(f"   {name}: {count:,}")
    
    def export_metrics(self, output_path: str, format: str = 'json') -> None:
        """Export metrics to file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        format : str
            Export format ('json' or 'csv')
        """
        try:
            self.metrics.export_metrics(output_path, format)
            print(f"✅ Metrics exported to {output_path}")
            
            # Show file info
            path_obj = Path(output_path)
            if path_obj.exists():
                size_mb = path_obj.stat().st_size / (1024 * 1024)
                print(f"📁 File size: {size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            sys.exit(1)
    
    def analyze_performance(self, operation: str = None) -> None:
        """Analyze performance patterns and provide insights.
        
        Parameters
        ----------
        operation : str, optional
            Focus analysis on specific operation
        """
        stats = self.metrics.get_summary_stats(operation=operation)
        
        print("🔍 Performance Analysis")
        print("=" * 50)
        
        # Overall health assessment
        issues = []
        recommendations = []
        
        # Memory analysis
        if 'memory' in stats:
            memory_percent = stats['memory']['current_percent']
            if memory_percent > 80:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                recommendations.append("Consider optimizing memory usage or increasing available RAM")
            elif memory_percent > 60:
                recommendations.append("Monitor memory usage trends")
        
        # Timing analysis
        if 'timing' in stats:
            timing = stats['timing']
            if timing['count'] > 0:
                avg_duration = timing['avg']
                max_duration = timing['max']
                
                if max_duration > avg_duration * 5:
                    issues.append(f"High timing variance: max={max_duration:.3f}s, avg={avg_duration:.3f}s")
                    recommendations.append("Investigate operations with high variance")
                
                if avg_duration > 1.0:
                    issues.append(f"Slow average performance: {avg_duration:.3f}s")
                    recommendations.append("Consider optimizing slow operations")
        
        # GPU analysis
        if 'gpu' in stats:
            gpu_util = stats['gpu']['utilization_percent']
            gpu_memory = stats['gpu']['memory_percent']
            gpu_temp = stats['gpu']['temperature_c']
            
            if gpu_util < 20:
                recommendations.append("Low GPU utilization - consider GPU acceleration opportunities")
            elif gpu_util > 95:
                issues.append(f"Very high GPU utilization: {gpu_util:.1f}%")
            
            if gpu_memory > 90:
                issues.append(f"High GPU memory usage: {gpu_memory:.1f}%")
                recommendations.append("Consider reducing batch sizes or model complexity")
            
            if gpu_temp > 80:
                issues.append(f"High GPU temperature: {gpu_temp:.1f}°C")
                recommendations.append("Check GPU cooling and reduce load if necessary")
        
        # Error analysis
        error_counters = {k: v for k, v in stats.get('counters', {}).items() if 'error' in k.lower() and v > 0}
        if error_counters:
            total_errors = sum(error_counters.values())
            issues.append(f"Errors detected: {total_errors} total")
            recommendations.append("Investigate error patterns and root causes")
        
        # Display results
        if issues:
            print("⚠️  Issues Detected:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("✅ No performance issues detected")
        
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Performance score
        score = 100
        score -= len(issues) * 15
        score -= len(error_counters) * 10
        score = max(0, score)
        
        print(f"\n🎯 Performance Score: {score}/100")
        
        if score >= 90:
            print("🌟 Excellent performance!")
        elif score >= 75:
            print("👍 Good performance")
        elif score >= 60:
            print("⚠️  Fair performance - room for improvement")
        else:
            print("🚨 Poor performance - optimization needed")
    
    def run(self):
        """Run the performance monitor CLI."""
        parser = argparse.ArgumentParser(
            description='Performance monitoring and metrics analysis',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Show live metrics with 2-second refresh
  python -m src.performance_monitor_cli --live --interval 2
  
  # Show summary for specific operation
  python -m src.performance_monitor_cli --summary --operation create_windows
  
  # Export metrics to JSON
  python -m src.performance_monitor_cli --export metrics.json
  
  # Analyze performance and get recommendations
  python -m src.performance_monitor_cli --analyze
            """
        )
        
        # Commands
        parser.add_argument('--live', action='store_true', help='Show live metrics')
        parser.add_argument('--summary', action='store_true', help='Show performance summary')
        parser.add_argument('--export', help='Export metrics to file')
        parser.add_argument('--analyze', action='store_true', help='Analyze performance and provide insights')
        
        # Options
        parser.add_argument('--operation', help='Filter by operation name')
        parser.add_argument('--interval', type=float, default=5.0, help='Refresh interval for live mode (seconds)')
        parser.add_argument('--iterations', type=int, help='Max iterations for live mode')
        parser.add_argument('--last-n', type=int, help='Show only last N entries')
        parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
        
        args = parser.parse_args()
        
        # Execute commands
        try:
            if args.live:
                self.show_live_metrics(args.interval, args.iterations)
            
            elif args.summary:
                self.show_summary(args.operation, args.last_n)
            
            elif args.export:
                self.export_metrics(args.export, args.format)
            
            elif args.analyze:
                self.analyze_performance(args.operation)
            
            else:
                # Default: show summary
                self.show_summary()
        
        except Exception as e:
            self.logger.error(f"CLI error: {e}")
            print(f"❌ Error: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    cli = PerformanceMonitorCLI()
    cli.run()


if __name__ == '__main__':
    main()
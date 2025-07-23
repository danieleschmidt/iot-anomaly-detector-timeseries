#!/usr/bin/env python3
"""
Cache Management CLI for IoT Anomaly Detection System

This CLI utility provides comprehensive cache management capabilities including
performance monitoring, statistics reporting, and cache optimization.
"""

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Dict, Any

from .caching_strategy import get_cache_stats, clear_all_caches
from .logging_config import get_logger


logger = get_logger(__name__)


def format_cache_stats(stats: Dict[str, Any], detailed: bool = False) -> str:
    """Format cache statistics for display."""
    if not stats:
        return "No cache statistics available"
    
    output = []
    output.append("=" * 60)
    output.append("CACHE PERFORMANCE STATISTICS")
    output.append("=" * 60)
    
    for cache_type, cache_stats in stats.items():
        if not cache_stats:
            continue
            
        output.append(f"\n{cache_type.upper()} CACHE:")
        output.append("-" * 40)
        
        hit_rate = cache_stats.get('hit_rate', 0) * 100
        total_requests = cache_stats.get('total_requests', 0)
        cache_size = cache_stats.get('cache_size', 0)
        max_size = cache_stats.get('max_size', 0)
        
        output.append(f"  Hit Rate:       {hit_rate:.1f}%")
        output.append(f"  Total Requests: {total_requests:,}")
        output.append(f"  Cache Hits:     {cache_stats.get('hits', 0):,}")
        output.append(f"  Cache Misses:   {cache_stats.get('misses', 0):,}")
        output.append(f"  Current Size:   {cache_size:,} / {max_size:,}")
        output.append(f"  Utilization:    {(cache_size/max_size*100) if max_size > 0 else 0:.1f}%")
        
        if detailed:
            output.append(f"  Memory Efficiency: {'High' if hit_rate > 70 else 'Medium' if hit_rate > 40 else 'Low'}")
            
            if total_requests > 0:
                savings_estimate = cache_stats.get('hits', 0) * 0.1  # Assume 100ms avg savings per hit
                output.append(f"  Est. Time Saved: {savings_estimate:.1f}s")
    
    output.append("\n" + "=" * 60)
    return "\n".join(output)


def analyze_cache_performance(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cache performance and provide recommendations."""
    analysis = {
        "overall_health": "good",
        "recommendations": [],
        "performance_score": 0,
        "issues": []
    }
    
    total_score = 0
    cache_count = 0
    
    for cache_type, cache_stats in stats.items():
        if not cache_stats:
            continue
            
        cache_count += 1
        hit_rate = cache_stats.get('hit_rate', 0) * 100
        utilization = (cache_stats.get('cache_size', 0) / cache_stats.get('max_size', 1)) * 100
        total_requests = cache_stats.get('total_requests', 0)
        
        # Score components (0-100)
        hit_score = min(hit_rate, 100)
        usage_score = min(total_requests / 10, 100)  # More requests = better
        efficiency_score = 100 if 20 <= utilization <= 80 else max(0, 100 - abs(50 - utilization))
        
        cache_score = (hit_score + usage_score + efficiency_score) / 3
        total_score += cache_score
        
        # Generate recommendations
        if hit_rate < 30:
            analysis["issues"].append(f"{cache_type} cache has low hit rate ({hit_rate:.1f}%)")
            analysis["recommendations"].append(f"Consider increasing {cache_type} cache size")
        elif hit_rate < 50:
            analysis["recommendations"].append(f"Monitor {cache_type} cache patterns for optimization")
        
        if utilization > 90:
            analysis["issues"].append(f"{cache_type} cache is nearly full ({utilization:.1f}%)")
            analysis["recommendations"].append(f"Increase {cache_type} cache maxsize")
        elif utilization < 10 and total_requests > 100:
            analysis["recommendations"].append(f"Consider reducing {cache_type} cache size")
        
        if total_requests < 10:
            analysis["recommendations"].append(f"Increase usage of {cache_type} operations for better performance")
    
    if cache_count > 0:
        analysis["performance_score"] = total_score / cache_count
        
        if analysis["performance_score"] >= 80:
            analysis["overall_health"] = "excellent"
        elif analysis["performance_score"] >= 60:
            analysis["overall_health"] = "good"
        elif analysis["performance_score"] >= 40:
            analysis["overall_health"] = "fair"
        else:
            analysis["overall_health"] = "poor"
    
    return analysis


def show_stats(detailed: bool = False, export_path: str = None) -> None:
    """Display cache statistics."""
    logger.info("Retrieving cache statistics...")
    
    try:
        stats = get_cache_stats()
        
        if not stats or all(not v for v in stats.values()):
            print("No cache activity detected. Caches may be empty or unused.")
            return
        
        # Display formatted statistics
        formatted_stats = format_cache_stats(stats, detailed=detailed)
        print(formatted_stats)
        
        if detailed:
            print("\nPERFORMANCE ANALYSIS:")
            print("-" * 40)
            analysis = analyze_cache_performance(stats)
            
            print(f"Overall Health: {analysis['overall_health'].upper()}")
            print(f"Performance Score: {analysis['performance_score']:.1f}/100")
            
            if analysis['issues']:
                print(f"\nIssues Found ({len(analysis['issues'])}):")
                for issue in analysis['issues']:
                    print(f"  ⚠️  {issue}")
            
            if analysis['recommendations']:
                print(f"\nRecommendations ({len(analysis['recommendations'])}):")
                for rec in analysis['recommendations']:
                    print(f"  💡 {rec}")
        
        # Export if requested
        if export_path:
            export_data = {
                "timestamp": time.time(),
                "stats": stats,
                "analysis": analyze_cache_performance(stats) if detailed else None
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"\nStatistics exported to: {export_path}")
            logger.info(f"Cache statistics exported to {export_path}")
    
    except Exception as e:
        logger.error(f"Failed to retrieve cache statistics: {e}")
        print(f"Error: {e}")


def clear_caches(confirm: bool = False) -> None:
    """Clear all caches."""
    if not confirm:
        response = input("Are you sure you want to clear all caches? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cache clear cancelled.")
            return
    
    try:
        logger.info("Clearing all caches...")
        clear_all_caches()
        print("✅ All caches cleared successfully.")
        logger.info("All caches cleared via CLI")
    
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        print(f"Error clearing caches: {e}")


def monitor_cache(interval: int = 5, duration: int = 60) -> None:
    """Monitor cache performance in real-time."""
    print(f"Starting cache monitoring (interval: {interval}s, duration: {duration}s)")
    print("Press Ctrl+C to stop monitoring early.\n")
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            stats = get_cache_stats()
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            print(f"REAL-TIME CACHE MONITORING")
            print(f"Elapsed: {time.time() - start_time:.1f}s / {duration}s")
            print(format_cache_stats(stats))
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    print("Cache monitoring completed.")


def benchmark_cache() -> None:
    """Run cache performance benchmark."""
    print("Running cache performance benchmark...")
    
    # This is a placeholder for actual benchmarking
    # In a real implementation, you might run specific operations
    # and measure cache performance
    
    import numpy as np
    from .data_preprocessor import DataPreprocessor
    
    try:
        # Create test data
        test_data = np.random.randn(1000, 10)
        preprocessor = DataPreprocessor()
        
        # Run operations multiple times to test caching
        print("Testing preprocessing cache performance...")
        
        start_time = time.time()
        for _ in range(3):
            windows = preprocessor.create_windows(test_data, window_size=30, step=1)
        first_run_time = time.time() - start_time
        
        # Show results
        stats = preprocessor.get_cache_stats()
        print(f"Benchmark completed in {first_run_time:.2f}s")
        print(format_cache_stats(stats['all_cache_stats']))
        
    except ImportError as e:
        print(f"Benchmark requires additional dependencies: {e}")
    except Exception as e:
        print(f"Benchmark failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cache Management CLI for IoT Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    stats_parser.add_argument(
        '--detailed', '-d', 
        action='store_true', 
        help='Show detailed analysis and recommendations'
    )
    stats_parser.add_argument(
        '--export', '-e', 
        type=str, 
        help='Export statistics to JSON file'
    )
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all caches')
    clear_parser.add_argument(
        '--yes', '-y', 
        action='store_true', 
        help='Skip confirmation prompt'
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor cache performance in real-time')
    monitor_parser.add_argument(
        '--interval', '-i', 
        type=int, 
        default=5, 
        help='Update interval in seconds (default: 5)'
    )
    monitor_parser.add_argument(
        '--duration', '-t', 
        type=int, 
        default=60, 
        help='Monitoring duration in seconds (default: 60)'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run cache performance benchmark')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'stats':
            show_stats(detailed=args.detailed, export_path=args.export)
        elif args.command == 'clear':
            clear_caches(confirm=args.yes)
        elif args.command == 'monitor':
            monitor_cache(interval=args.interval, duration=args.duration)
        elif args.command == 'benchmark':
            benchmark_cache()
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"CLI command failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
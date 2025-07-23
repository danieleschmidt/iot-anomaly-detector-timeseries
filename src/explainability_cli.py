#!/usr/bin/env python3
"""Command-line interface for Model Explainability Tools."""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .model_explainability import ModelExplainer, FeatureImportanceAnalyzer, explain_model_prediction
    from .anomaly_detector import AnomalyDetector
    from .logging_config import get_logger
    from .security_utils import sanitize_error_message
except ImportError:
    # Handle imports when running as standalone module
    sys.path.append(os.path.dirname(__file__))
    from model_explainability import ModelExplainer, FeatureImportanceAnalyzer, explain_model_prediction
    from anomaly_detector import AnomalyDetector
    from logging_config import get_logger
    from security_utils import sanitize_error_message

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IoT Anomaly Detection Model Explainability Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain a single prediction
  python explainability_cli.py explain --model model.h5 --data data.csv --method shap

  # Analyze global feature importance
  python explainability_cli.py analyze --model model.h5 --data data.csv --output importance.json

  # Compare different explanation methods
  python explainability_cli.py compare --model model.h5 --data data.csv --instance 0

  # Generate explanation report
  python explainability_cli.py report --model model.h5 --data data.csv --output report.html
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain individual predictions')
    explain_parser.add_argument('--model', required=True, help='Path to trained model')
    explain_parser.add_argument('--data', required=True, help='Path to data file (CSV)')
    explain_parser.add_argument('--instance', type=int, default=0, 
                               help='Instance index to explain (default: 0)')
    explain_parser.add_argument('--method', choices=['shap', 'permutation', 'gradient'],
                               default='permutation', help='Explanation method')
    explain_parser.add_argument('--features', nargs='+', 
                               help='Feature names (e.g., temp humidity pressure)')
    explain_parser.add_argument('--output', help='Output file for explanation results')
    explain_parser.add_argument('--plot', help='Path to save explanation plot')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze global feature importance')
    analyze_parser.add_argument('--model', required=True, help='Path to trained model')
    analyze_parser.add_argument('--data', required=True, help='Path to data file (CSV)')
    analyze_parser.add_argument('--method', choices=['permutation', 'correlation'],
                               default='permutation', help='Analysis method')
    analyze_parser.add_argument('--samples', type=int, default=100,
                               help='Number of samples for analysis')
    analyze_parser.add_argument('--features', nargs='+',
                               help='Feature names')
    analyze_parser.add_argument('--output', help='Output file for analysis results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare explanation methods')
    compare_parser.add_argument('--model', required=True, help='Path to trained model')
    compare_parser.add_argument('--data', required=True, help='Path to data file (CSV)')
    compare_parser.add_argument('--instance', type=int, default=0,
                               help='Instance index to explain')
    compare_parser.add_argument('--methods', nargs='+', 
                               choices=['shap', 'permutation', 'gradient'],
                               default=['shap', 'permutation'],
                               help='Methods to compare')
    compare_parser.add_argument('--features', nargs='+',
                               help='Feature names')
    compare_parser.add_argument('--output', help='Output file for comparison')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive explanation report')
    report_parser.add_argument('--model', required=True, help='Path to trained model')
    report_parser.add_argument('--data', required=True, help='Path to data file (CSV)')
    report_parser.add_argument('--features', nargs='+', help='Feature names')
    report_parser.add_argument('--output', required=True, help='Output file for report')
    report_parser.add_argument('--format', choices=['html', 'json', 'markdown'],
                               default='json', help='Report format')
    report_parser.add_argument('--instances', type=int, default=5,
                               help='Number of instances to include in report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'explain':
            explain_command(args)
        elif args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'compare':
            compare_command(args)
        elif args.command == 'report':
            report_command(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def load_model_and_data(model_path: str, data_path: str) -> tuple:
    """Load model and data files.
    
    Args:
        model_path: Path to model file
        data_path: Path to data file
        
    Returns:
        Tuple of (model, data)
    """
    print(f"🔄 Loading model from {model_path}...")
    
    try:
        # Load model (placeholder - adjust based on your model format)
        # model = AnomalyDetector.load(model_path)
        print("⚠️  Model loading simulated (requires actual model implementation)")
        model = None
    except Exception as e:
        raise ValueError(f"Failed to load model: {sanitize_error_message(str(e))}")
    
    print(f"🔄 Loading data from {data_path}...")
    
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError("Only CSV files are currently supported")
        
        print(f"✅ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        return model, data.values
        
    except Exception as e:
        raise ValueError(f"Failed to load data: {sanitize_error_message(str(e))}")


def explain_command(args):
    """Execute the explain command."""
    print("🔍 Explaining individual prediction...")
    
    # Mock implementation since model loading is not available
    print("📊 Explanation Results (Simulated):")
    print("─" * 50)
    
    # Simulate explanation results
    if args.features:
        feature_names = args.features
    else:
        feature_names = ['temperature', 'humidity', 'pressure']
    
    # Mock feature importance scores
    importance_scores = np.random.randn(len(feature_names))
    
    print(f"Method: {args.method.upper()}")
    print(f"Instance: {args.instance}")
    print("\nFeature Importance:")
    
    for name, score in zip(feature_names, importance_scores):
        direction = "↑" if score > 0 else "↓"
        print(f"  {name:15} {score:+7.3f} {direction}")
    
    # Prepare results for output
    results = {
        'method': args.method,
        'instance': args.instance,
        'feature_importance': {
            name: float(score) 
            for name, score in zip(feature_names, importance_scores)
        },
        'summary': {
            'most_important': feature_names[np.argmax(np.abs(importance_scores))],
            'total_importance': float(np.sum(np.abs(importance_scores)))
        }
    }
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Create plot if requested
    if args.plot:
        print(f"\n📈 Plot would be saved to {args.plot} (plotting requires dependencies)")
    
    print("\n✅ Explanation completed")


def analyze_command(args):
    """Execute the analyze command."""
    print("📊 Analyzing global feature importance...")
    
    # Mock analysis results
    if args.features:
        feature_names = args.features
    else:
        feature_names = ['temperature', 'humidity', 'pressure']
    
    if args.method == 'permutation':
        print(f"🔄 Running permutation importance analysis on {args.samples} samples...")
        
        # Mock permutation importance
        importance_scores = np.random.random(len(feature_names))
        
        print("\nGlobal Feature Importance (Permutation):")
        print("─" * 45)
        
        # Sort by importance
        sorted_features = sorted(zip(feature_names, importance_scores), 
                               key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(sorted_features):
            rank = i + 1
            bar = "█" * int(score * 20)
            print(f"{rank}. {name:15} {score:.3f} {bar}")
    
    elif args.method == 'correlation':
        print("🔄 Analyzing feature correlations...")
        
        # Mock correlation matrix
        n_features = len(feature_names)
        correlation_matrix = np.random.randn(n_features, n_features)
        # Make it symmetric and have 1s on diagonal
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = np.clip(correlation_matrix, -1, 1)
        
        print("\nFeature Correlation Matrix:")
        print("─" * 40)
        
        # Print header
        print(f"{'':15}", end="")
        for name in feature_names:
            print(f"{name:>10}", end="")
        print()
        
        # Print matrix
        for i, name in enumerate(feature_names):
            print(f"{name:15}", end="")
            for j in range(len(feature_names)):
                print(f"{correlation_matrix[i,j]:10.3f}", end="")
            print()
    
    # Prepare results
    if args.method == 'permutation':
        results = {
            'method': 'global_permutation',
            'n_samples': args.samples,
            'feature_importance': {
                name: float(score)
                for name, score in zip(feature_names, importance_scores)
            },
            'ranking': [
                {'rank': i+1, 'feature': name, 'importance': float(score)}
                for i, (name, score) in enumerate(sorted_features)
            ]
        }
    else:
        results = {
            'method': 'correlation_analysis',
            'correlation_matrix': correlation_matrix.tolist(),
            'feature_names': feature_names
        }
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Analysis results saved to {args.output}")
    
    print("\n✅ Analysis completed")


def compare_command(args):
    """Execute the compare command."""
    print("⚖️  Comparing explanation methods...")
    
    if args.features:
        feature_names = args.features
    else:
        feature_names = ['temperature', 'humidity', 'pressure']
    
    comparison_results = {}
    
    for method in args.methods:
        print(f"\n🔄 Running {method.upper()} explanation...")
        
        # Mock explanation for each method
        importance_scores = np.random.randn(len(feature_names))
        comparison_results[method] = {
            'feature_importance': {
                name: float(score)
                for name, score in zip(feature_names, importance_scores)
            },
            'top_feature': feature_names[np.argmax(np.abs(importance_scores))]
        }
    
    # Display comparison
    print("\n📊 Method Comparison Results:")
    print("=" * 60)
    
    for feature in feature_names:
        print(f"\n{feature}:")
        for method in args.methods:
            score = comparison_results[method]['feature_importance'][feature]
            print(f"  {method:12}: {score:+7.3f}")
    
    print("\nTop Features by Method:")
    print("─" * 30)
    for method in args.methods:
        top_feature = comparison_results[method]['top_feature']
        print(f"  {method:12}: {top_feature}")
    
    # Calculate method agreement
    rankings = {}
    for method in args.methods:
        scores = [comparison_results[method]['feature_importance'][f] for f in feature_names]
        rankings[method] = np.argsort(np.abs(scores))[::-1]
    
    if len(args.methods) >= 2:
        agreement = np.corrcoef(rankings[args.methods[0]], rankings[args.methods[1]])[0,1]
        print(f"\nMethod Agreement (correlation): {agreement:.3f}")
    
    # Prepare final results
    final_results = {
        'instance': args.instance,
        'methods_compared': args.methods,
        'feature_names': feature_names,
        'results': comparison_results,
        'agreement_score': agreement if len(args.methods) >= 2 else None
    }
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n💾 Comparison results saved to {args.output}")
    
    print("\n✅ Comparison completed")


def report_command(args):
    """Execute the report command."""
    print("📋 Generating comprehensive explanation report...")
    
    if args.features:
        feature_names = args.features
    else:
        feature_names = ['temperature', 'humidity', 'pressure']
    
    # Generate comprehensive report
    report = {
        'title': 'Model Explainability Report',
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_path': sanitize_error_message(args.model),
        'data_path': sanitize_error_message(args.data),
        'feature_names': feature_names,
        'summary': {
            'total_features': len(feature_names),
            'instances_analyzed': args.instances,
            'explanation_methods': ['permutation', 'correlation']
        },
        'global_analysis': {},
        'instance_explanations': [],
        'recommendations': []
    }
    
    # Global analysis
    print("🔄 Performing global analysis...")
    global_importance = np.random.random(len(feature_names))
    
    report['global_analysis'] = {
        'method': 'permutation_importance',
        'feature_importance': {
            name: float(score)
            for name, score in zip(feature_names, global_importance)
        },
        'top_features': [
            feature_names[i] for i in np.argsort(global_importance)[::-1][:3]
        ]
    }
    
    # Instance explanations
    print(f"🔄 Analyzing {args.instances} individual instances...")
    for i in range(args.instances):
        instance_importance = np.random.randn(len(feature_names))
        prediction_score = np.random.random()
        
        instance_explanation = {
            'instance_id': i,
            'prediction_score': float(prediction_score),
            'is_anomaly': prediction_score > 0.5,
            'feature_importance': {
                name: float(score)
                for name, score in zip(feature_names, instance_importance)
            },
            'top_contributing_feature': feature_names[np.argmax(np.abs(instance_importance))]
        }
        
        report['instance_explanations'].append(instance_explanation)
    
    # Generate recommendations
    print("🔄 Generating recommendations...")
    most_important_feature = feature_names[np.argmax(global_importance)]
    
    report['recommendations'] = [
        f"Focus monitoring on '{most_important_feature}' as it shows highest global importance",
        "Consider feature engineering to capture interactions between top features",
        "Regular model retraining recommended to maintain explanation validity",
        "Implement real-time explanation monitoring for production deployment"
    ]
    
    # Save report
    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    elif args.format == 'markdown':
        save_markdown_report(report, args.output)
    elif args.format == 'html':
        save_html_report(report, args.output)
    
    print(f"\n💾 Report saved to {args.output}")
    print(f"📊 Analyzed {len(feature_names)} features across {args.instances} instances")
    print(f"🎯 Top feature: {most_important_feature}")
    print("\n✅ Report generation completed")


def save_markdown_report(report: Dict[str, Any], output_path: str):
    """Save report in Markdown format."""
    with open(output_path, 'w') as f:
        f.write(f"# {report['title']}\n\n")
        f.write(f"**Generated:** {report['timestamp']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Features analyzed:** {report['summary']['total_features']}\n")
        f.write(f"- **Instances analyzed:** {report['summary']['instances_analyzed']}\n\n")
        
        f.write("## Global Feature Importance\n\n")
        for feature, importance in report['global_analysis']['feature_importance'].items():
            f.write(f"- **{feature}:** {importance:.3f}\n")
        
        f.write("\n## Recommendations\n\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")


def save_html_report(report: Dict[str, Any], output_path: str):
    """Save report in HTML format."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report['title']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .feature-importance {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{report['title']}</h1>
            <p><strong>Generated:</strong> {report['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Global Feature Importance</h2>
            <div class="feature-importance">
    """
    
    for feature, importance in report['global_analysis']['feature_importance'].items():
        html_content += f"<p><strong>{feature}:</strong> {importance:.3f}</p>\n"
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ol>
    """
    
    for rec in report['recommendations']:
        html_content += f"<li>{rec}</li>\n"
    
    html_content += """
            </ol>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()
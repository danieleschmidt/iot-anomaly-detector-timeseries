#!/usr/bin/env python3
"""Command-line utility for managing model versions and metadata."""

import argparse
import json
import sys
from pathlib import Path

from .model_metadata import ModelMetadata


class ModelManager:
    """
    Model management class for handling model versions and metadata.
    
    This class provides a programmatic interface to model management functionality
    that's used by integration tests and other parts of the system.
    """
    
    def __init__(self, model_directory: str = "saved_models"):
        """
        Initialize ModelManager.
        
        Parameters
        ----------
        model_directory : str
            Directory where models and metadata are stored
        """
        self.model_directory = Path(model_directory)
        self.metadata_manager = ModelMetadata(model_directory)
    
    def list_models(self) -> list:
        """
        List all available model versions.
        
        Returns
        -------
        list
            List of model version information dictionaries
        """
        return self.metadata_manager.list_model_versions()
    
    def get_model_metadata(self, version: str) -> dict:
        """
        Get metadata for a specific model version.
        
        Parameters
        ----------
        version : str
            Model version identifier
            
        Returns
        -------
        dict
            Model metadata
        """
        metadata_path = self.model_directory / f"metadata_{version}.json"
        return self.metadata_manager.load_metadata(metadata_path)
    
    def compare_models(self, version1: str, version2: str) -> dict:
        """
        Compare two model versions.
        
        Parameters
        ----------
        version1 : str
            First model version
        version2 : str
            Second model version
            
        Returns
        -------
        dict
            Comparison results
        """
        meta1 = self.get_model_metadata(version1)
        meta2 = self.get_model_metadata(version2)
        
        return {
            "version1": version1,
            "version2": version2,
            "metadata1": meta1,
            "metadata2": meta2
        }
    
    def cleanup_old_models(self, keep_count: int = 5) -> list:
        """
        Clean up old model versions.
        
        Parameters
        ----------
        keep_count : int
            Number of recent models to keep
            
        Returns
        -------
        list
            List of removed model files
        """
        versions = self.list_models()
        if len(versions) <= keep_count:
            return []
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Remove old versions
        removed_files = []
        for version_info in versions[keep_count:]:
            model_file = self.model_directory / version_info['model_file']
            metadata_file = self.model_directory / f"metadata_{version_info['version']}.json"
            
            if model_file.exists():
                model_file.unlink()
                removed_files.append(str(model_file))
            
            if metadata_file.exists():
                metadata_file.unlink()
                removed_files.append(str(metadata_file))
        
        return removed_files


def list_models(args):
    """List all available model versions."""
    metadata_manager = ModelMetadata(args.model_directory)
    versions = metadata_manager.list_model_versions()
    
    if not versions:
        print("No model versions found.")
        return
    
    print(f"Found {len(versions)} model versions:")
    print("-" * 80)
    print(f"{'Version':<20} {'Created':<20} {'Epochs':<10} {'Model File':<30}")
    print("-" * 80)
    
    for version_info in versions:
        created = version_info['created_at'][:19].replace('T', ' ')  # Format datetime
        epochs = str(version_info['training_epochs'])
        model_file = version_info['model_file']
        
        print(f"{version_info['version']:<20} {created:<20} {epochs:<10} {model_file:<30}")
    
    print("-" * 80)


def show_metadata(args):
    """Show detailed metadata for a specific model version."""
    metadata_manager = ModelMetadata(args.model_directory)
    metadata_path = Path(args.model_directory) / f"metadata_{args.version}.json"
    
    try:
        metadata = metadata_manager.load_metadata(metadata_path)
        print(json.dumps(metadata, indent=2))
    except FileNotFoundError:
        print(f"Error: Model version '{args.version}' not found.")
        sys.exit(1)


def compare_models(args):
    """Compare two model versions."""
    metadata_manager = ModelMetadata(args.model_directory)
    
    try:
        comparison = metadata_manager.compare_models(args.version1, args.version2)
        
        print(f"Comparing {args.version1} vs {args.version2}")
        print("=" * 50)
        
        print("\\nParameter Differences:")
        if comparison['parameter_differences']:
            for param, diff in comparison['parameter_differences'].items():
                print(f"  {param}: {diff['version1']} -> {diff['version2']}")
        else:
            print("  No parameter differences found.")
        
        print("\\nPerformance Differences:")
        if comparison['performance_differences']:
            for metric, diff in comparison['performance_differences'].items():
                print(f"  {metric}: {diff['version1']} -> {diff['version2']}")
        else:
            print("  No performance differences found.")
        
        print(f"\\nModel Hash Different: {comparison['model_hash_different']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cleanup_old_models(args):
    """Remove old model versions, keeping only the N most recent."""
    metadata_manager = ModelMetadata(args.model_directory)
    versions = metadata_manager.list_model_versions()
    
    if len(versions) <= args.keep:
        print(f"Only {len(versions)} versions found, nothing to clean up.")
        return
    
    versions_to_delete = versions[args.keep:]
    
    print(f"Will delete {len(versions_to_delete)} old model versions:")
    for version_info in versions_to_delete:
        print(f"  - {version_info['version']} ({version_info['created_at'][:19]})")
    
    if not args.force:
        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    model_dir = Path(args.model_directory)
    deleted_count = 0
    
    for version_info in versions_to_delete:
        version = version_info['version']
        
        # Delete metadata file
        metadata_file = model_dir / f"metadata_{version}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            print(f"Deleted metadata: {metadata_file}")
        
        # Delete model file (if it exists and is in the same directory)
        model_file = model_dir / version_info['model_file']
        if model_file.exists():
            model_file.unlink()
            print(f"Deleted model: {model_file}")
        
        deleted_count += 1
    
    print(f"Cleanup complete. Deleted {deleted_count} model versions.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage IoT Anomaly Detector model versions and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.model_manager list
  python -m src.model_manager show v20240120_143022
  python -m src.model_manager compare v20240120_143022 v20240121_091234
  python -m src.model_manager cleanup --keep 5 --force
        """
    )
    
    parser.add_argument(
        "--model-directory", 
        default="saved_models",
        help="Directory containing models and metadata (default: saved_models)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all model versions")
    list_parser.set_defaults(func=list_models)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show metadata for a specific version")
    show_parser.add_argument("version", help="Model version to show")
    show_parser.set_defaults(func=show_metadata)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument("version1", help="First model version")
    compare_parser.add_argument("version2", help="Second model version")
    compare_parser.set_defaults(func=compare_models)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old model versions")
    cleanup_parser.add_argument(
        "--keep", 
        type=int, 
        default=5, 
        help="Number of recent versions to keep (default: 5)"
    )
    cleanup_parser.add_argument(
        "--force", 
        action="store_true", 
        help="Skip confirmation prompt"
    )
    cleanup_parser.set_defaults(func=cleanup_old_models)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Command-line utility for managing flexible autoencoder architectures."""

import argparse
import json
import sys
from pathlib import Path

from .flexible_autoencoder import (
    get_predefined_architectures,
    load_architecture_config,
    create_autoencoder_from_config
)


def list_architectures(args):
    """List all available predefined architectures."""
    architectures = get_predefined_architectures()
    
    print(f"Available predefined architectures ({len(architectures)}):")
    print("=" * 60)
    
    for name, config in architectures.items():
        description = config.get('description', 'No description available')
        input_shape = config.get('input_shape', 'Unknown')
        latent_dim = config.get('latent_config', {}).get('dim', 'Unknown')
        num_layers = len(config.get('encoder_layers', []))
        
        print(f"Name: {name}")
        print(f"  Description: {description}")
        print(f"  Input Shape: {input_shape}")
        print(f"  Latent Dim: {latent_dim}")
        print(f"  Encoder Layers: {num_layers}")
        
        if args.verbose:
            print("  Layers:")
            for i, layer in enumerate(config.get('encoder_layers', [])):
                layer_type = layer.get('type', 'unknown')
                layer_params = {k: v for k, v in layer.items() if k != 'type'}
                print(f"    {i+1}. {layer_type}: {layer_params}")
        
        print()


def show_architecture(args):
    """Show detailed information about a specific architecture."""
    architectures = get_predefined_architectures()
    
    if args.name not in architectures:
        available = list(architectures.keys())
        print(f"Error: Architecture '{args.name}' not found.")
        print(f"Available architectures: {', '.join(available)}")
        sys.exit(1)
    
    config = architectures[args.name]
    
    print(f"Architecture: {args.name}")
    print("=" * 50)
    print(json.dumps(config, indent=2))


def validate_config(args):
    """Validate an architecture configuration file."""
    config_path = Path(args.config_file)
    
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config_file}' not found.")
        sys.exit(1)
    
    try:
        config = load_architecture_config(config_path)
        print(f"✓ Configuration file '{args.config_file}' is valid.")
        
        if args.verbose:
            print("\\nConfiguration details:")
            print(json.dumps(config, indent=2))
            
    except Exception as e:
        print(f"✗ Configuration file '{args.config_file}' is invalid: {e}")
        sys.exit(1)


def create_template(args):
    """Create a template architecture configuration file."""
    if args.architecture:
        # Create template based on predefined architecture
        architectures = get_predefined_architectures()
        if args.architecture not in architectures:
            available = list(architectures.keys())
            print(f"Error: Architecture '{args.architecture}' not found.")
            print(f"Available architectures: {', '.join(available)}")
            sys.exit(1)
        
        config = architectures[args.architecture].copy()
        config['name'] = f"custom_{args.architecture}"
        config['description'] = f"Custom architecture based on {args.architecture}"
        
    else:
        # Create minimal template
        config = {
            "name": "custom_architecture",
            "description": "Custom autoencoder architecture",
            "input_shape": [30, 3],
            "encoder_layers": [
                {
                    "type": "lstm",
                    "units": 64,
                    "return_sequences": True,
                    "dropout": 0.1
                },
                {
                    "type": "batch_norm"
                },
                {
                    "type": "lstm",
                    "units": 32,
                    "return_sequences": False,
                    "dropout": 0.1
                }
            ],
            "latent_config": {
                "dim": 16,
                "activation": "linear",
                "regularization": "l2"
            },
            "compilation": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"]
            }
        }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Template configuration saved to '{args.output}'")
    print("You can now edit this file to customize the architecture.")


def test_architecture(args):
    """Test building an architecture from configuration."""
    config_path = Path(args.config_file)
    
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config_file}' not found.")
        sys.exit(1)
    
    try:
        # Load and validate configuration
        config = load_architecture_config(config_path)
        print("✓ Configuration loaded successfully.")
        
        # Test building the model
        model = create_autoencoder_from_config(config)
        
        if model is not None:
            print("✓ Model built successfully!")
            print(f"  - Total parameters: {model.count_params():,}")
            print(f"  - Input shape: {config['input_shape']}")
            print(f"  - Latent dimension: {config['latent_config']['dim']}")
            print(f"  - Number of layers: {len(model.layers)}")
            
            if args.verbose:
                print("\\nModel summary:")
                model.summary()
        else:
            print("✓ Model configuration is valid (TensorFlow not available for actual building)")
            
    except Exception as e:
        print(f"✗ Error building model: {e}")
        sys.exit(1)


def compare_architectures(args):
    """Compare two architecture configurations."""
    # Load architectures
    if args.arch1 in get_predefined_architectures():
        config1 = get_predefined_architectures()[args.arch1]
        source1 = f"predefined ({args.arch1})"
    else:
        config1 = load_architecture_config(args.arch1)
        source1 = f"file ({args.arch1})"
    
    if args.arch2 in get_predefined_architectures():
        config2 = get_predefined_architectures()[args.arch2]
        source2 = f"predefined ({args.arch2})"
    else:
        config2 = load_architecture_config(args.arch2)
        source2 = f"file ({args.arch2})"
    
    print("Comparing architectures:")
    print(f"  Architecture 1: {source1}")
    print(f"  Architecture 2: {source2}")
    print("=" * 60)
    
    # Compare basic properties
    print("Basic Properties:")
    print(f"  Input Shape: {config1.get('input_shape')} vs {config2.get('input_shape')}")
    print(f"  Latent Dim: {config1.get('latent_config', {}).get('dim')} vs {config2.get('latent_config', {}).get('dim')}")
    print(f"  Encoder Layers: {len(config1.get('encoder_layers', []))} vs {len(config2.get('encoder_layers', []))}")
    
    # Compare layer types
    layers1 = [layer.get('type') for layer in config1.get('encoder_layers', [])]
    layers2 = [layer.get('type') for layer in config2.get('encoder_layers', [])]
    
    print("\\nLayer Types:")
    print(f"  Architecture 1: {' -> '.join(layers1)}")
    print(f"  Architecture 2: {' -> '.join(layers2)}")
    
    # Compare compilation settings
    comp1 = config1.get('compilation', {})
    comp2 = config2.get('compilation', {})
    
    print("\\nCompilation:")
    print(f"  Optimizer: {comp1.get('optimizer')} vs {comp2.get('optimizer')}")
    print(f"  Loss: {comp1.get('loss')} vs {comp2.get('loss')}")
    
    if args.verbose:
        print("\\nFull Configurations:")
        print("\\nArchitecture 1:")
        print(json.dumps(config1, indent=2))
        print("\\nArchitecture 2:")
        print(json.dumps(config2, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage flexible autoencoder architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all predefined architectures
  python -m src.architecture_manager list
  
  # Show details of a specific architecture
  python -m src.architecture_manager show simple_lstm
  
  # Create a template configuration file
  python -m src.architecture_manager template -o my_arch.json
  
  # Create template based on existing architecture
  python -m src.architecture_manager template -a deep_lstm -o custom_deep.json
  
  # Validate a configuration file
  python -m src.architecture_manager validate -c my_arch.json
  
  # Test building a model from configuration
  python -m src.architecture_manager test -c my_arch.json
  
  # Compare two architectures
  python -m src.architecture_manager compare simple_lstm deep_lstm
        """
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List predefined architectures")
    list_parser.set_defaults(func=list_architectures)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show architecture details")
    show_parser.add_argument("name", help="Architecture name")
    show_parser.set_defaults(func=show_architecture)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("-c", "--config-file", required=True, help="Configuration file path")
    validate_parser.set_defaults(func=validate_config)
    
    # Template command
    template_parser = subparsers.add_parser("template", help="Create template configuration")
    template_parser.add_argument("-o", "--output", default="architecture_template.json", 
                                help="Output file path")
    template_parser.add_argument("-a", "--architecture", 
                                help="Base template on predefined architecture")
    template_parser.set_defaults(func=create_template)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test building model from configuration")
    test_parser.add_argument("-c", "--config-file", required=True, help="Configuration file path")
    test_parser.set_defaults(func=test_architecture)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two architectures")
    compare_parser.add_argument("arch1", help="First architecture (name or file path)")
    compare_parser.add_argument("arch2", help="Second architecture (name or file path)")
    compare_parser.set_defaults(func=compare_architectures)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
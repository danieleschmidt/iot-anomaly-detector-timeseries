#!/usr/bin/env python3
"""Command-line interface for the Model Serving API."""

import argparse
import sys
import os
from pathlib import Path

try:
    from .model_serving_api import FASTAPI_AVAILABLE
    from .logging_config import get_logger
except ImportError:
    # Handle imports when running as standalone module
    sys.path.append(os.path.dirname(__file__))
    from model_serving_api import FASTAPI_AVAILABLE
    from logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IoT Anomaly Detection Model Serving API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python model_api_cli.py start

  # Start server with custom host and port
  python model_api_cli.py start --host 0.0.0.0 --port 8080

  # Start server with pre-loaded model
  python model_api_cli.py start --model path/to/model.h5

  # Check if API dependencies are available
  python model_api_cli.py check

  # Show API server status
  python model_api_cli.py status --url http://localhost:8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start server command
    start_parser = subparsers.add_parser('start', help='Start the API server')
    start_parser.add_argument('--host', default='0.0.0.0',  # nosec B104
                             help='Host to bind to (default: 0.0.0.0) - 0.0.0.0 required for containers/cloud')
    start_parser.add_argument('--port', type=int, default=8000,
                             help='Port to bind to (default: 8000)')
    start_parser.add_argument('--model', type=str,
                             help='Path to model file to pre-load')
    start_parser.add_argument('--reload', action='store_true',
                             help='Enable auto-reload for development')
    
    # Check dependencies command
    subparsers.add_parser('check', help='Check API dependencies')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check server status')
    status_parser.add_argument('--url', default='http://localhost:8000',
                              help='API server URL (default: http://localhost:8000)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run API tests')
    test_parser.add_argument('--url', default='http://localhost:8000',
                            help='API server URL to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'start':
        start_server_command(args)
    elif args.command == 'check':
        check_dependencies()
    elif args.command == 'status':
        check_server_status(args.url)
    elif args.command == 'test':
        test_api_endpoints(args.url)


def start_server_command(args):
    """Start the API server."""
    print("🚀 Starting IoT Anomaly Detection API Server...")
    
    if not FASTAPI_AVAILABLE:
        print("❌ Error: FastAPI and uvicorn are required to start the server.")
        print("   Please install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    # Validate model path if provided
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Error: Model file not found: {args.model}")
            sys.exit(1)
        print(f"📂 Pre-loading model: {args.model}")
    
    print(f"🌐 Server will start on http://{args.host}:{args.port}")
    print(f"📖 API documentation: http://{args.host}:{args.port}/docs")
    print("📊 Health check: http://{args.host}:{args.port}/health")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Import uvicorn here to show better error message
        import uvicorn
        from model_serving_api import app, model_server
        
        # Pre-load model if specified
        if args.model:
            success = model_server.load_model(args.model)
            if success:
                print("✅ Model loaded successfully")
            else:
                print("⚠️  Warning: Failed to load model, server will start without model")
        
        # Start the server
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
        
    except ImportError:
        print("❌ Error: uvicorn is required to start the server.")
        print("   Please install with: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking API dependencies...\n")
    
    dependencies = {
        'FastAPI': ('fastapi', 'pip install fastapi'),
        'Uvicorn': ('uvicorn', 'pip install uvicorn'),
        'Pydantic': ('pydantic', 'pip install pydantic'),
        'Requests': ('requests', 'pip install requests'),
    }
    
    all_available = True
    
    for name, (module, install_cmd) in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Not available - {install_cmd}")
            all_available = False
    
    print(f"\n{'✅ All dependencies available!' if all_available else '❌ Some dependencies missing'}")
    
    if not all_available:
        print("\n📦 Install all API dependencies with:")
        print("   pip install fastapi uvicorn pydantic requests")


def check_server_status(url: str):
    """Check if the API server is running."""
    print(f"🔍 Checking server status at {url}...")
    
    try:
        import requests
        
        # Check health endpoint
        response = requests.get(f"{url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Uptime: {data.get('uptime_seconds', 0):.1f} seconds")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            
    except ImportError:
        print("❌ Error: 'requests' library is required for status check.")
        print("   Please install with: pip install requests")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {url}")
        print("   Make sure the server is running")
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout connecting to {url}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")


def test_api_endpoints(url: str):
    """Test API endpoints with sample data."""
    print(f"🧪 Testing API endpoints at {url}...")
    
    try:
        import requests
        
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
        
        # Test model info endpoint
        print("2. Testing model info endpoint...")
        response = requests.get(f"{url}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Model info: {info.get('model_name')}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
        
        # Test metrics endpoint
        print("3. Testing metrics endpoint...")
        response = requests.get(f"{url}/model/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print(f"   ✅ Metrics: {metrics.get('total_predictions')} predictions")
        else:
            print(f"   ❌ Metrics failed: {response.status_code}")
        
        # Test prediction endpoint with sample data
        print("4. Testing prediction endpoint...")
        sample_data = {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "window_size": 2
        }
        
        response = requests.post(
            f"{url}/model/predict",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful: {len(result.get('anomaly_scores', []))} scores")
        elif response.status_code == 503:
            print("   ⚠️  Prediction failed: No model loaded (expected)")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"      Response: {response.text}")
        
        print("\n✅ API endpoint tests completed")
        
    except ImportError:
        print("❌ Error: 'requests' library is required for testing.")
        print("   Please install with: pip install requests")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == '__main__':
    main()
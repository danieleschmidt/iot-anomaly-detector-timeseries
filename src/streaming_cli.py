#!/usr/bin/env python3
"""Command-line interface for streaming anomaly detection."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any

from .streaming_processor import StreamingProcessor, StreamingConfig
from .logging_config import get_logger
from .security_utils import secure_json_load, sanitize_error_message


class StreamingCLI:
    """Command-line interface for streaming anomaly detection."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.processor = None
        self.running = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal, stopping streaming...")
        self.running = False
        if self.processor:
            self.processor.stop_streaming()
        sys.exit(0)
    
    def anomaly_callback(self, result: Dict[str, Any]) -> None:
        """Default anomaly detection callback."""
        anomaly_count = result.get('anomaly_count', 0)
        timestamp = result.get('timestamp', 'unknown')
        
        print(f"🚨 ANOMALY ALERT: {anomaly_count} anomalies detected at {timestamp}")
        
        # Log detailed information
        scores = result.get('scores', [])
        anomalies = result.get('anomalies', [])
        
        for i, (score, is_anomaly) in enumerate(zip(scores, anomalies)):
            if is_anomaly:
                print(f"   Window {i}: Score = {score:.4f}")
    
    def run_interactive_mode(self, processor: StreamingProcessor) -> None:
        """Run interactive streaming mode."""
        print("\n🔄 Starting interactive streaming mode")
        print("📡 Waiting for data input...")
        print("💡 Type 'help' for commands, 'quit' to exit")
        
        processor.start_streaming()
        self.running = True
        
        while self.running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'help':
                    self.print_interactive_help()
                
                elif user_input.lower() == 'status':
                    metrics = processor.get_performance_metrics()
                    self.print_metrics(metrics)
                
                elif user_input.lower().startswith('data '):
                    # Manual data input: data sensor1:1.5,sensor2:2.3
                    data_str = user_input[5:].strip()
                    try:
                        data_point = self.parse_data_input(data_str)
                        processor.ingest_data(data_point)
                        print(f"✅ Ingested data point: {data_point}")
                    except Exception as e:
                        print(f"❌ Error parsing data: {e}")
                
                elif user_input.lower() == 'detect':
                    # Manual anomaly detection trigger
                    result = processor.detect_anomalies()
                    if result.get('scores'):
                        anomaly_count = result.get('anomaly_count', 0)
                        print(f"🔍 Detection complete: {anomaly_count} anomalies found")
                    else:
                        print("⚠️ Insufficient data for detection")
                
                else:
                    print(f"❌ Unknown command: {user_input}")
                    print("💡 Type 'help' for available commands")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
        
        print("\n🛑 Stopping streaming mode...")
        processor.stop_streaming()
    
    def print_interactive_help(self) -> None:
        """Print help for interactive mode."""
        print("""
📖 Interactive Mode Commands:
  help           - Show this help message
  status         - Show performance metrics
  data <values>  - Add data point (e.g., 'data sensor1:1.5,sensor2:2.3')
  detect         - Manually trigger anomaly detection
  quit           - Exit streaming mode
        """)
    
    def parse_data_input(self, data_str: str) -> Dict[str, Any]:
        """Parse manual data input."""
        data_point = {'timestamp': time.time()}
        
        for pair in data_str.split(','):
            key, value = pair.strip().split(':')
            data_point[key.strip()] = float(value.strip())
        
        return data_point
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print performance metrics."""
        print(f"""
📊 Performance Metrics:
  📈 Total Processed: {metrics['total_processed']}
  🚨 Anomalies Detected: {metrics['anomalies_detected']}
  ⚡ Processing Rate: {metrics['processing_rate']:.2f} windows/sec
  📊 Anomaly Rate: {metrics['anomaly_rate']:.3%}
  💾 Buffer Utilization: {metrics['buffer_utilization']:.1%}
  ⏱️ Elapsed Time: {metrics['elapsed_time']:.1f} seconds
  ▶️ Running: {'Yes' if metrics['is_running'] else 'No'}
        """)
    
    def run(self):
        """Run the streaming CLI."""
        parser = argparse.ArgumentParser(
            description='Real-time IoT anomaly detection streaming processor',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start interactive streaming mode
  python -m src.streaming_cli --model models/autoencoder.h5 --interactive
  
  # Process data from file with custom config
  python -m src.streaming_cli --model models/autoencoder.h5 --input data.json --config streaming_config.json
  
  # Export results after processing
  python -m src.streaming_cli --model models/autoencoder.h5 --input data.json --output results.json
            """
        )
        
        parser.add_argument('--model', required=True, help='Path to trained autoencoder model')
        parser.add_argument('--scaler', help='Path to trained scaler (optional)')
        parser.add_argument('--config', help='Path to streaming configuration JSON file')
        parser.add_argument('--input', help='Path to input data file (JSON format)')
        parser.add_argument('--output', help='Path to output results file')
        parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
        
        # Configuration parameters
        parser.add_argument('--window-size', type=int, default=50, help='Window size for sequences')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
        parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
        parser.add_argument('--buffer-size', type=int, default=1000, help='Streaming buffer size')
        parser.add_argument('--interval', type=float, default=1.0, help='Processing interval in seconds')
        
        args = parser.parse_args()
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Load or create configuration
            if args.config:
                try:
                    config_dict = secure_json_load(args.config, max_size_mb=1.0)
                    config = StreamingConfig.from_dict(config_dict)
                    self.logger.info(f"Loaded configuration from {sanitize_error_message(args.config)}")
                except Exception as e:
                    sanitized_error = sanitize_error_message(str(e))
                    self.logger.error(f"Failed to load configuration: {sanitized_error}")
                    raise
            else:
                config = StreamingConfig(
                    window_size=args.window_size,
                    batch_size=args.batch_size,
                    anomaly_threshold=args.threshold,
                    buffer_size=args.buffer_size,
                    processing_interval=args.interval
                )
            
            # Initialize streaming processor
            self.processor = StreamingProcessor(
                model_path=args.model,
                config=config,
                scaler_path=args.scaler
            )
            
            # Add anomaly callback
            self.processor.add_anomaly_callback(self.anomaly_callback)
            
            print(f"🚀 Streaming processor initialized")
            print(f"📊 Config: Window={config.window_size}, Threshold={config.anomaly_threshold}")
            
            # Run based on mode
            if args.interactive:
                self.run_interactive_mode(self.processor)
            
            elif args.input:
                self.process_input_file(args.input, args.output)
            
            else:
                print("❌ Error: Must specify either --interactive or --input")
                sys.exit(1)
        
        except Exception as e:
            self.logger.error(f"Error in streaming CLI: {e}")
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def process_input_file(self, input_path: str, output_path: str = None) -> None:
        """Process data from input file."""
        self.logger.info(f"Processing input file: {sanitize_error_message(input_path)}")
        
        try:
            # Use secure JSON loading with size limits
            data = secure_json_load(input_path, max_size_mb=50.0)
            
            if isinstance(data, dict) and 'data' in data:
                data_points = data['data']
            elif isinstance(data, list):
                data_points = data
            else:
                raise ValueError("Input file must contain a list of data points or dict with 'data' key")
            
            print(f"📁 Processing {len(data_points)} data points...")
            
            # Start streaming processing
            self.processor.start_streaming()
            
            # Ingest data points
            for i, data_point in enumerate(data_points):
                self.processor.ingest_data(data_point)
                
                if (i + 1) % 100 == 0:
                    print(f"📈 Processed {i + 1}/{len(data_points)} points...")
                
                # Small delay to simulate streaming
                time.sleep(0.01)
            
            # Final detection
            print("🔍 Running final anomaly detection...")
            final_result = self.processor.detect_anomalies()
            
            # Stop streaming
            self.processor.stop_streaming()
            
            # Show final metrics
            metrics = self.processor.get_performance_metrics()
            self.print_metrics(metrics)
            
            # Export results if requested
            if output_path:
                self.processor.export_results(output_path)
                print(f"💾 Results exported to {output_path}")
            
            print("✅ Processing complete!")
            
        except Exception as e:
            self.logger.error(f"Error processing input file: {e}")
            raise


def main():
    """Main entry point."""
    cli = StreamingCLI()
    cli.run()


if __name__ == '__main__':
    main()
# Changelog

## v0.0.8 - Real-Time Streaming Processing

### Added
- **Complete Streaming Data Processing System** (`streaming_processor.py`)
  - Real-time anomaly detection for continuous IoT data streams
  - Configurable circular buffer management with automatic overflow handling
  - Threading-based continuous processing with graceful shutdown mechanisms
  - Comprehensive callback system for immediate anomaly notifications and alerts
  - Production-ready performance monitoring with detailed metrics collection
- **StreamingConfig Dataclass** with automatic validation and serialization
  - Configurable window sizes, batch sizes, and anomaly thresholds
  - Buffer size and processing interval customization
  - JSON serialization support for configuration persistence
- **Enhanced DataPreprocessor** with streaming-optimized methods
  - New `create_sliding_windows()` method for memory-efficient real-time processing
  - Automatic scaler fitting and transformation for streaming data
  - Optimized for continuous data ingestion without memory leaks
- **Comprehensive Streaming CLI Interface** (`streaming_cli.py`)
  - Interactive mode for real-time data input and monitoring
  - Batch file processing with progress tracking and status updates
  - Configurable processing parameters via command line or JSON configuration
  - Real-time performance metrics display and anomaly alerts
  - Results export in multiple formats (JSON, CSV) for analysis
- **Comprehensive Test Suite** (`test_streaming_processor.py`)
  - 15+ test cases covering all streaming functionality and edge cases
  - Mock-based testing for isolated unit testing without external dependencies
  - Configuration validation and serialization testing
  - Performance metrics and callback system validation
  - Cross-platform compatibility testing

### Technical Features
- **Thread-Safe Operations**: Proper resource cleanup and thread management
- **Memory Efficiency**: Circular buffer prevents memory growth with continuous data
- **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM for production deployment
- **Error Handling**: Comprehensive error handling with detailed logging and recovery
- **Performance Monitoring**: Real-time metrics including processing rate, anomaly rate, buffer utilization
- **Callback System**: Extensible callback architecture for custom anomaly handling
- **Configuration Management**: Flexible configuration with validation and defaults

### Integration & Compatibility  
- **Seamless Integration**: Works with existing models, preprocessing, and infrastructure
- **Backward Compatibility**: Existing batch processing functionality unchanged
- **CLI Integration**: Consistent command-line interface with existing tools
- **Production Ready**: Comprehensive logging, monitoring, and error handling

### Use Cases Enabled
- **Real-Time IoT Monitoring**: Continuous anomaly detection for sensor data streams
- **Interactive Analysis**: Manual data input and real-time anomaly detection
- **Batch Stream Processing**: Process large datasets as if they were streaming
- **Production Deployment**: Thread-safe streaming with monitoring and alerts

### Documentation Updates
- Updated BACKLOG.md with completed streaming implementation and new priority queue
- Enhanced technical debt log with resolved streaming requirements
- Added comprehensive feature documentation and usage examples

## v0.0.7 - Performance Optimization: Batched Inference

### Added
- **Batched Inference Processing** (`score_batched()` method in AnomalyDetector)
  - Memory-efficient batch processing for large datasets (10-100x larger capacity)
  - Configurable batch size with optimal default (256 sequences per batch)
  - Progress logging every 10 batches for long-running operations
  - Identical accuracy to original `score()` method with improved scalability
- **Automatic Smart Batching** in `predict()` method
  - Automatic detection and batching for datasets >1000 windows  
  - Optional force batching for smaller datasets via `use_batched` parameter
  - Backward compatibility maintained with existing code
- **Enhanced CLI Interface** 
  - New `--batch-size` option for fine-tuned memory control
  - New `--use-batched` flag to force batched processing
  - Automatic optimization recommendations in logging
- **Comprehensive Performance Test Suite** (`test_anomaly_detector_performance.py`)
  - 15+ test cases covering batching functionality and edge cases
  - Memory usage validation and performance benchmarking
  - Cross-platform compatibility testing with mocked dependencies

### Performance Improvements  
- **Memory Efficiency**: Reduces peak memory usage by processing in chunks
- **Scalability**: Enables processing of datasets 10-100x larger than before
- **Progress Feedback**: Real-time progress logging for long operations
- **Backward Compatibility**: Existing code continues to work without changes
- **Smart Optimization**: Automatic batching recommendation for large datasets

### Technical Improvements
- Enhanced error handling for edge cases (empty datasets, single sequences)
- Comprehensive logging with performance metrics and progress tracking
- Memory-conscious batch processing with configurable chunk sizes
- Robust validation testing with extensive mock-based unit tests

### Documentation Updates
- Updated BACKLOG.md with completed performance optimization
- Enhanced CLI help text with batching options and recommendations
- Added comprehensive docstrings with usage examples

## v0.0.6 - Comprehensive Test Coverage Enhancement

### Added
- **Complete Test Suite for Model Management** (`test_model_manager.py`)
  - 50+ test cases covering all CLI commands (list, show, compare, cleanup)
  - Comprehensive error handling and edge case validation
  - File operation safety testing with permission handling
  - User interaction simulation and confirmation testing
  - Mock-based testing for isolated unit testing
- **Complete Test Suite for Architecture Management** (`test_architecture_manager.py`)
  - 40+ test cases covering flexible architecture system
  - Architecture validation and comparison logic testing
  - Template creation and configuration loading validation
  - CLI command testing with comprehensive mocking
  - Integration testing with FlexibleAutoencoderBuilder
- **Complete Test Suite for Autoencoder Model** (`test_autoencoder_model.py`)
  - 25+ test cases covering model building and validation
  - Model structure and parameter validation testing
  - Integration testing with complete training workflows
  - Performance and memory usage testing
  - TensorFlow availability graceful handling
- **Enhanced Test Coverage Metrics**
  - Added 115+ comprehensive test cases total
  - Covered previously untested critical modules
  - Systematic edge case and error condition testing
  - Cross-platform compatibility testing

### Technical Improvements
- Eliminated potential data loss from model management operations through comprehensive testing
- Improved code reliability through systematic edge case validation
- Enhanced debugging capabilities with thorough error handling tests
- Strengthened production readiness through comprehensive CLI testing
- Added graceful handling for environments without TensorFlow
- Implemented mock-based testing for improved test isolation

### Documentation Updates
- Updated BACKLOG.md with completed test coverage improvements
- Added detailed test impact analysis and risk reduction metrics
- Updated technical debt log with resolved test coverage issues

## v0.0.5 - Data Validation & Quality Assurance

### Added
- **Comprehensive Data Validation Module** (`data_validator.py`)
  - Three validation levels: Strict, Moderate, Permissive
  - File format validation (existence, readability, size checks)
  - Schema validation (column presence, data types, structure)
  - Data quality checks (missing values, duplicates, constant columns, outliers)
  - Time series validation (sequence length, monotonicity, intervals)
  - Auto-fix capabilities for common data issues
- **CLI Interface for Data Validation**
  - Standalone validation tool with extensive options
  - Support for expected columns, time column specification
  - Auto-fix mode with cleaned data output
  - Detailed validation reports in Markdown and JSON formats
  - Verbose mode for comprehensive analysis
- **DataPreprocessor Integration**
  - Automatic validation in preprocessing pipeline
  - Configurable validation levels and auto-fix options
  - Backward compatibility with existing code
  - Enhanced error reporting with validation details
- **Comprehensive Test Suite**
  - 50+ test cases covering all validation scenarios
  - Unit tests for all validator components
  - Integration tests for complete validation pipeline
  - Edge case testing and error handling validation
- **Documentation Updates**
  - Detailed README section on data validation usage
  - CLI examples and integration patterns
  - Updated backlog with completed validation tasks

### Technical Improvements
- Enhanced error handling in data preprocessing
- Structured validation results with detailed reporting
- Configurable validation strictness for different use cases
- Performance-optimized validation checks
- Memory-efficient handling of large datasets

## v0.0.4

- Added MIT license and contributor guide
- Introduced setup script and CI workflow
- README documents new setup steps

## v0.0.3

- Add quantile-based thresholding option to evaluation script
- Document new CLI usage and mutually exclusive flags
- Validate quantile bounds in `evaluate_model`

## v0.0.2

- Add quantile-based thresholding for anomaly detection and CLI
- Clarify mutual exclusivity between `--threshold` and `--quantile`
- Early validation of the `--quantile` argument in the CLI

## v0.0.1

- Applying previous commit.
- Merge pull request #3 from danieleschmidt/codex/continue-development-per-plan
- Parameterize anomaly injection in data generation
- Merge pull request #2 from danieleschmidt/codex/continue-development-per-plan
- Add CLI for anomaly detection
- Merge pull request #1 from danieleschmidt/codex/create-development-plan-checklist-in-markdown
- Enhance evaluation script with CLI and JSON output
- Update README.md
- Initial commit

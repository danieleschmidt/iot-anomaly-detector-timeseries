# Changelog

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

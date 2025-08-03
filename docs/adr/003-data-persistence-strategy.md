# ADR-003: Data Persistence Strategy

## Status
Accepted

## Context
The IoT Anomaly Detection system needs to persist various types of data:
- Time series sensor data (high volume, write-heavy)
- Model artifacts (large binary files)
- Configuration data (small, frequently read)
- Metrics and logs (append-only, time-based)
- Anomaly detection results (structured, queryable)

## Decision
We will use a polyglot persistence strategy with specialized storage solutions:

### 1. Time Series Data
**Technology**: InfluxDB / TimescaleDB
- Optimized for time series workloads
- Efficient compression and retention policies
- Built-in downsampling and aggregation
- Native time-based queries

### 2. Model Artifacts
**Technology**: Object Storage (S3-compatible)
- Versioned storage for model files
- Cost-effective for large binary data
- Integration with ML frameworks
- Lifecycle management policies

### 3. Application Data
**Technology**: PostgreSQL
- Configuration management
- User preferences
- Anomaly metadata
- System state

### 4. Cache Layer
**Technology**: Redis
- Model inference caching
- Session management
- Real-time metrics buffering
- Distributed locks

### 5. Message Queue
**Technology**: Apache Kafka / RabbitMQ
- Streaming data ingestion
- Event-driven processing
- Reliable message delivery
- Horizontal scalability

## Implementation Strategy

### Repository Pattern
```python
class BaseRepository:
    def create(self, entity)
    def read(self, id)
    def update(self, entity)
    def delete(self, id)
    def find_by(self, criteria)

class TimeSeriesRepository(BaseRepository):
    def write_points(self, points)
    def query_range(self, start, end)
    def downsample(self, interval)

class ModelRepository(BaseRepository):
    def save_model(self, model, metadata)
    def load_model(self, version)
    def list_versions(self)
```

### Connection Management
- Connection pooling for all databases
- Retry logic with exponential backoff
- Circuit breakers for fault tolerance
- Health checks and monitoring

## Consequences

### Positive
- Optimal storage for each data type
- Scalability for different workloads
- Cost optimization
- Performance optimization

### Negative
- Operational complexity
- Multiple technologies to maintain
- Data consistency challenges
- Backup and recovery complexity

### Mitigation
- Unified monitoring and alerting
- Automated backup strategies
- Transaction coordination where needed
- Comprehensive documentation
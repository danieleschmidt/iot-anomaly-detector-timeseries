#!/usr/bin/env python3
"""
Enterprise API Gateway for IoT Anomaly Detection System
Provides authentication, rate limiting, request routing, and comprehensive middleware.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import jwt
from pydantic import BaseModel, Field
import uvicorn

from .config import get_config
from .logging_config import setup_logging
from .health_monitoring import HealthMonitor


# Request/Response Models
class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    data: List[List[float]] = Field(..., description="Time series data array")
    window_size: Optional[int] = Field(30, description="Sliding window size")
    threshold: Optional[float] = Field(None, description="Anomaly threshold")
    quantile: Optional[float] = Field(None, description="Threshold quantile")


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""
    anomalies: List[int] = Field(..., description="Anomaly flags (0/1)")
    scores: List[float] = Field(..., description="Anomaly scores")
    threshold: float = Field(..., description="Applied threshold")
    processing_time_ms: float = Field(..., description="Processing time")
    request_id: str = Field(..., description="Request identifier")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    checks: Dict[str, Any]


class RateLimitInfo(BaseModel):
    """Rate limiting information"""
    limit: int
    remaining: int
    reset_time: int


# Rate Limiter Implementation
class TokenBucketRateLimiter:
    """Token bucket rate limiter for API requests"""
    
    def __init__(self, rate: int, capacity: int, window_seconds: int = 60):
        self.rate = rate  # tokens per window
        self.capacity = capacity  # max tokens
        self.window_seconds = window_seconds
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'tokens': capacity,
                'last_update': time.time(),
                'request_times': deque()
            }
        )
        
    def _get_client_key(self, request: Request) -> str:
        """Generate unique key for client identification"""
        # Use IP address and User-Agent for client identification
        client_ip = request.client.host
        user_agent = request.headers.get("User-Agent", "")
        return hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()
    
    def _refill_bucket(self, bucket: Dict[str, Any]) -> None:
        """Refill token bucket based on elapsed time"""
        now = time.time()
        elapsed = now - bucket['last_update']
        
        if elapsed > 0:
            tokens_to_add = (elapsed / self.window_seconds) * self.rate
            bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_update'] = now
    
    def _cleanup_old_requests(self, bucket: Dict[str, Any]) -> None:
        """Remove old request timestamps"""
        cutoff = time.time() - self.window_seconds
        while bucket['request_times'] and bucket['request_times'][0] < cutoff:
            bucket['request_times'].popleft()
    
    def is_allowed(self, request: Request) -> Tuple[bool, RateLimitInfo]:
        """Check if request is allowed under rate limit"""
        client_key = self._get_client_key(request)
        bucket = self.buckets[client_key]
        
        self._refill_bucket(bucket)
        self._cleanup_old_requests(bucket)
        
        now = time.time()
        
        # Check token bucket
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            bucket['request_times'].append(now)
            
            rate_limit_info = RateLimitInfo(
                limit=self.rate,
                remaining=int(bucket['tokens']),
                reset_time=int(now + self.window_seconds)
            )
            return True, rate_limit_info
        else:
            rate_limit_info = RateLimitInfo(
                limit=self.rate,
                remaining=0,
                reset_time=int(bucket['last_update'] + self.window_seconds)
            )
            return False, rate_limit_info


# Authentication Handler
class JWTAuthHandler:
    """JWT-based authentication handler"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                credentials.credentials, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


# API Gateway Class
class APIGateway:
    """Enterprise API Gateway implementation"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging(__name__)
        self.health_monitor = HealthMonitor()
        
        # Initialize rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            rate=self.config.security.rate_limit_per_minute,
            capacity=self.config.security.rate_limit_per_minute * 2
        )
        
        # Initialize authentication if enabled
        self.auth_handler = None
        if self.config.security.enable_authentication:
            if not self.config.security.jwt_secret:
                raise ValueError("JWT secret required when authentication is enabled")
            self.auth_handler = JWTAuthHandler(self.config.security.jwt_secret)
        
        # Request tracking
        self.request_counter = 0
        self.request_times = deque()
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan management"""
            self.logger.info("API Gateway starting up...")
            yield
            self.logger.info("API Gateway shutting down...")
        
        app = FastAPI(
            title="IoT Anomaly Detection API Gateway",
            description="Enterprise-grade API gateway for IoT anomaly detection services",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add middleware
        self._add_middleware(app)
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_middleware(self, app: FastAPI) -> None:
        """Add middleware to FastAPI app"""
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        if self.config.security.enable_ssl:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        
        # Request logging middleware
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Rate limiting
            allowed, rate_info = self.rate_limiter.is_allowed(request)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "rate_limit": rate_info.dict()
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_info.limit),
                        "X-RateLimit-Remaining": str(rate_info.remaining),
                        "X-RateLimit-Reset": str(rate_info.reset_time)
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_info.reset_time)
            
            # Log request
            process_time = time.time() - start_time
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} time={process_time:.3f}s id={request_id}"
            )
            
            # Track metrics
            self.request_counter += 1
            self.request_times.append(time.time())
            
            return response
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes"""
        
        def get_current_user():
            """Dependency for authentication"""
            if self.auth_handler:
                return Depends(self.auth_handler.verify_token)
            return None
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check endpoint"""
            try:
                health_report = await self.health_monitor.run_all_checks()
                
                return HealthResponse(
                    status=health_report.overall_status.value,
                    timestamp=health_report.timestamp.isoformat(),
                    uptime_seconds=health_report.uptime_seconds,
                    version="1.0.0",
                    checks={
                        check.name: {
                            "status": check.status.value,
                            "message": check.message,
                            "duration_ms": check.duration_ms
                        } for check in health_report.checks
                    }
                )
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Health check failed"
                )
        
        @app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint"""
            now = time.time()
            
            # Clean old request times (last 5 minutes)
            cutoff = now - 300
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            
            request_rate = len(self.request_times) / 300  # requests per second
            
            metrics = {
                "total_requests": self.request_counter,
                "request_rate_per_second": request_rate,
                "active_connections": len(self.rate_limiter.buckets),
                "uptime_seconds": now - self.health_monitor.start_time
            }
            
            return metrics
        
        @app.post("/api/v1/detect", response_model=AnomalyDetectionResponse)
        async def detect_anomalies(
            request_data: AnomalyDetectionRequest,
            request: Request,
            current_user=get_current_user()
        ):
            """Detect anomalies in time series data"""
            start_time = time.time()
            request_id = request.state.request_id
            
            try:
                # Import here to avoid circular imports
                from .anomaly_detector import AnomalyDetector
                
                # Initialize detector
                detector = AnomalyDetector(
                    "saved_models/autoencoder.h5",
                    "saved_models/scaler.pkl"
                )
                
                # Convert input data to required format
                import pandas as pd
                import numpy as np
                
                data_array = np.array(request_data.data)
                df = pd.DataFrame(data_array)
                
                # Detect anomalies
                if request_data.threshold:
                    anomaly_flags = detector.predict_dataframe(
                        df, 
                        threshold=request_data.threshold,
                        window_size=request_data.window_size
                    )
                elif request_data.quantile:
                    anomaly_flags = detector.predict_dataframe(
                        df,
                        quantile=request_data.quantile, 
                        window_size=request_data.window_size
                    )
                else:
                    anomaly_flags = detector.predict_dataframe(
                        df,
                        window_size=request_data.window_size
                    )
                
                # Get reconstruction scores
                scores = detector.get_reconstruction_errors(df, window_size=request_data.window_size)
                threshold = detector.threshold if hasattr(detector, 'threshold') else 0.5
                
                processing_time = (time.time() - start_time) * 1000
                
                return AnomalyDetectionResponse(
                    anomalies=anomaly_flags.tolist(),
                    scores=scores.tolist(),
                    threshold=threshold,
                    processing_time_ms=processing_time,
                    request_id=request_id
                )
                
            except Exception as e:
                self.logger.error(f"Anomaly detection failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Anomaly detection failed: {str(e)}"
                )
        
        @app.post("/api/v1/auth/token")
        async def create_token(username: str, password: str):
            """Create JWT token for authentication"""
            if not self.auth_handler:
                raise HTTPException(
                    status_code=501,
                    detail="Authentication not enabled"
                )
            
            # Simple credential check (replace with proper authentication)
            if username == "admin" and password == "admin123":
                token_data = {"sub": username}
                expires = timedelta(hours=self.config.security.jwt_expiry_hours)
                token = self.auth_handler.create_access_token(token_data, expires)
                
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": self.config.security.jwt_expiry_hours * 3600
                }
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API gateway"""
        self.logger.info(f"Starting API Gateway on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


# CLI Interface
async def main():
    """Main entry point for API Gateway"""
    gateway = APIGateway()
    
    # Run the gateway
    config = gateway.config
    gateway.run(
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=config.security.ssl_key_path if config.security.enable_ssl else None,
        ssl_certfile=config.security.ssl_cert_path if config.security.enable_ssl else None
    )


if __name__ == "__main__":
    asyncio.run(main())
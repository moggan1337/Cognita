"""
Metrics Tracking Module

Provides utilities for tracking and logging federated learning metrics.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time

import numpy as np


@dataclass
class MetricRecord:
    """Single metric record."""
    name: str
    value: float
    timestamp: float
    round: Optional[int] = None
    client_id: Optional[str] = None


class MetricsTracker:
    """Track metrics for federated learning experiments.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.log("accuracy", 0.95, round=1)
        >>> tracker.log("loss", 0.05, round=1)
        >>> 
        >>> summary = tracker.get_summary()
        >>> print(summary)
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self._metrics: Dict[str, List[MetricRecord]] = {}
        self._start_time = time.time()
        
    def log(
        self,
        name: str,
        value: float,
        round: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> None:
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            round: Optional round number
            client_id: Optional client identifier
        """
        if name not in self._metrics:
            self._metrics[name] = []
            
        record = MetricRecord(
            name=name,
            value=value,
            timestamp=time.time(),
            round=round,
            client_id=client_id,
        )
        self._metrics[name].append(record)
        
    def get(
        self,
        name: str,
        rounds: Optional[List[int]] = None,
    ) -> List[float]:
        """Get metric values.
        
        Args:
            name: Metric name
            rounds: Optional filter by rounds
            
        Returns:
            List of metric values
        """
        if name not in self._metrics:
            return []
            
        records = self._metrics[name]
        
        if rounds:
            records = [r for r in records if r.round in rounds]
            
        return [r.value for r in records]
        
    def get_mean(
        self,
        name: str,
        rounds: Optional[List[int]] = None,
    ) -> float:
        """Get mean metric value.
        
        Args:
            name: Metric name
            rounds: Optional filter by rounds
            
        Returns:
            Mean value
        """
        values = self.get(name, rounds)
        return np.mean(values) if values else 0.0
        
    def get_latest(
        self,
        name: str,
    ) -> Optional[float]:
        """Get latest metric value.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        return self._metrics[name][-1].value
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for name, records in self._metrics.items():
            values = [r.value for r in records]
            summary[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "latest": values[-1] if values else None,
                "count": len(values),
            }
            
        summary["elapsed_time"] = time.time() - self._start_time
        
        return summary
        
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {}
        self._start_time = time.time()

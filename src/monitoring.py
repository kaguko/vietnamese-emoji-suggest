"""
Model monitoring module for Vietnamese Emoji Suggestion System.

This module provides:
- Prediction logging
- Performance tracking
- Weekly/monthly reports
- Drift detection
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class PredictionLog:
    """Single prediction log entry."""
    timestamp: str
    input_text: str
    predicted_emotion: str
    predicted_intensity: float
    suggested_emojis: List[str]
    confidence: float
    user_feedback: Optional[str] = None  # "positive", "negative", None
    selected_emoji: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class DailyMetrics:
    """Daily aggregated metrics."""
    date: str
    total_predictions: int
    avg_confidence: float
    avg_latency_ms: float
    positive_feedback_rate: float
    emotion_distribution: Dict[str, int]
    top_emojis: List[str]


# Evaluation targets from feedback
EVALUATION_TARGETS = {
    "emotion_accuracy": 0.70,      # 70% emotion classification accuracy
    "intensity_mse": 0.5,          # MSE for intensity prediction
    "user_satisfaction": 0.75,     # 75% positive feedback rate
    "inference_time": 0.3,         # Max 300ms inference time
    "precision_at_3": 0.62,        # 62% precision@3
    "recall_at_5": 0.75,           # 75% recall@5
    "mrr": 0.65,                   # Mean Reciprocal Rank
    "ndcg_at_5": 0.70              # NDCG@5
}


class ModelMonitor:
    """
    Monitor model performance and log predictions.
    
    Features:
    - Real-time prediction logging
    - Performance metric tracking
    - Weekly/monthly report generation
    - Drift detection alerts
    """
    
    def __init__(
        self,
        log_path: str = "data/logs/predictions.jsonl",
        metrics_path: str = "data/logs/metrics.json",
        alert_threshold: float = 0.15  # Alert if metrics drop by 15%
    ):
        """
        Initialize monitor.
        
        Args:
            log_path: Path for prediction logs (JSONL format)
            metrics_path: Path for aggregated metrics
            alert_threshold: Threshold for drift detection alerts
        """
        self.log_path = log_path
        self.metrics_path = metrics_path
        self.alert_threshold = alert_threshold
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # In-memory buffer for batch writing
        self.log_buffer: List[PredictionLog] = []
        self.buffer_size = 10
        
        # Load existing metrics
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load metrics from storage."""
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {'daily': {}, 'weekly': {}, 'alerts': []}
    
    def _save_metrics(self):
        """Save metrics to storage."""
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
    
    def _flush_buffer(self):
        """Write buffered logs to file."""
        if not self.log_buffer:
            return
            
        with open(self.log_path, 'a', encoding='utf-8') as f:
            for log in self.log_buffer:
                f.write(json.dumps(asdict(log), ensure_ascii=False) + '\n')
        
        self.log_buffer = []
    
    def log_prediction(
        self,
        input_text: str,
        predicted_emotion: str,
        predicted_intensity: float,
        suggested_emojis: List[str],
        confidence: float,
        latency_ms: float = 0.0
    ) -> str:
        """
        Log a single prediction.
        
        Args:
            input_text: Input text (may be truncated for privacy)
            predicted_emotion: Predicted emotion
            predicted_intensity: Predicted intensity (0-1)
            suggested_emojis: List of suggested emojis
            confidence: Model confidence score
            latency_ms: Inference latency in milliseconds
            
        Returns:
            Log entry ID (timestamp)
        """
        # Truncate input for privacy
        truncated_input = input_text[:50] + "..." if len(input_text) > 50 else input_text
        
        log_entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            input_text=truncated_input,
            predicted_emotion=predicted_emotion,
            predicted_intensity=predicted_intensity,
            suggested_emojis=suggested_emojis,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
        self.log_buffer.append(log_entry)
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            self._flush_buffer()
        
        return log_entry.timestamp
    
    def record_feedback(
        self,
        timestamp: str,
        feedback_type: str,
        selected_emoji: Optional[str] = None
    ):
        """
        Record user feedback for a prediction.
        
        Args:
            timestamp: Original prediction timestamp
            feedback_type: "positive" or "negative"
            selected_emoji: Emoji selected by user (if any)
        """
        # Update in buffer if present
        for log in self.log_buffer:
            if log.timestamp == timestamp:
                log.user_feedback = feedback_type
                log.selected_emoji = selected_emoji
                return
        
        # TODO: Update in file (would require more complex storage)
    
    def compute_daily_metrics(self, date_str: Optional[str] = None) -> DailyMetrics:
        """
        Compute metrics for a specific day.
        
        Args:
            date_str: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            DailyMetrics object
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Flush buffer first
        self._flush_buffer()
        
        # Read logs for the day
        day_logs = []
        
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log = json.loads(line)
                        if log['timestamp'].startswith(date_str):
                            day_logs.append(log)
                    except json.JSONDecodeError:
                        continue
        
        if not day_logs:
            return DailyMetrics(
                date=date_str,
                total_predictions=0,
                avg_confidence=0.0,
                avg_latency_ms=0.0,
                positive_feedback_rate=0.0,
                emotion_distribution={},
                top_emojis=[]
            )
        
        # Compute metrics
        confidences = [log['confidence'] for log in day_logs]
        latencies = [log['latency_ms'] for log in day_logs]
        
        # Emotion distribution
        emotion_dist = defaultdict(int)
        for log in day_logs:
            emotion_dist[log['predicted_emotion']] += 1
        
        # Top emojis
        emoji_counts = defaultdict(int)
        for log in day_logs:
            for emoji in log['suggested_emojis'][:3]:
                emoji_counts[emoji] += 1
        top_emojis = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Feedback rate
        feedback_logs = [log for log in day_logs if log.get('user_feedback')]
        positive_rate = 0.0
        if feedback_logs:
            positive_count = sum(1 for log in feedback_logs if log['user_feedback'] == 'positive')
            positive_rate = positive_count / len(feedback_logs)
        
        metrics = DailyMetrics(
            date=date_str,
            total_predictions=len(day_logs),
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            positive_feedback_rate=positive_rate,
            emotion_distribution=dict(emotion_dist),
            top_emojis=[e for e, _ in top_emojis]
        )
        
        # Store in metrics
        self.metrics['daily'][date_str] = asdict(metrics)
        self._save_metrics()
        
        return metrics
    
    def generate_weekly_report(self) -> Dict:
        """
        Generate weekly performance report.
        
        Returns:
            Dict with weekly statistics and trends
        """
        today = datetime.now()
        week_start = today - timedelta(days=7)
        
        report = {
            'period': f"{week_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}",
            'generated_at': today.isoformat(),
            'summary': {},
            'daily_breakdown': [],
            'trends': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Collect daily metrics
        daily_data = []
        for i in range(7):
            date = (week_start + timedelta(days=i+1)).strftime('%Y-%m-%d')
            metrics = self.compute_daily_metrics(date)
            daily_data.append(metrics)
            report['daily_breakdown'].append(asdict(metrics))
        
        # Compute summary
        total_predictions = sum(m.total_predictions for m in daily_data)
        if total_predictions > 0:
            report['summary'] = {
                'total_predictions': total_predictions,
                'avg_daily_predictions': total_predictions / 7,
                'avg_confidence': statistics.mean([m.avg_confidence for m in daily_data if m.avg_confidence > 0] or [0]),
                'avg_latency_ms': statistics.mean([m.avg_latency_ms for m in daily_data if m.avg_latency_ms > 0] or [0]),
                'avg_satisfaction': statistics.mean([m.positive_feedback_rate for m in daily_data if m.positive_feedback_rate > 0] or [0])
            }
            
            # Check against targets
            for metric, target in EVALUATION_TARGETS.items():
                if metric == 'user_satisfaction':
                    actual = report['summary']['avg_satisfaction']
                    if actual < target:
                        report['alerts'].append({
                            'metric': metric,
                            'target': target,
                            'actual': actual,
                            'gap': target - actual
                        })
                elif metric == 'inference_time':
                    actual = report['summary']['avg_latency_ms'] / 1000  # Convert to seconds
                    if actual > target:
                        report['alerts'].append({
                            'metric': metric,
                            'target': target,
                            'actual': actual,
                            'issue': 'latency_exceeded'
                        })
            
            # Recommendations
            if report['alerts']:
                report['recommendations'].append(
                    "Review model performance - some metrics below target"
                )
            if total_predictions < 100:
                report['recommendations'].append(
                    "Consider collecting more user data for better analysis"
                )
        
        # Store weekly report
        week_key = f"week_{today.strftime('%Y_%W')}"
        self.metrics['weekly'][week_key] = report
        self._save_metrics()
        
        return report
    
    def check_drift(self) -> List[Dict]:
        """
        Check for performance drift.
        
        Returns:
            List of drift alerts
        """
        alerts = []
        
        today = datetime.now()
        current_date = today.strftime('%Y-%m-%d')
        previous_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        
        current = self.metrics['daily'].get(current_date, {})
        previous = self.metrics['daily'].get(previous_date, {})
        
        if not current or not previous:
            return alerts
        
        # Check confidence drift
        if previous.get('avg_confidence', 0) > 0:
            conf_change = (current.get('avg_confidence', 0) - previous['avg_confidence']) / previous['avg_confidence']
            if abs(conf_change) > self.alert_threshold:
                alerts.append({
                    'type': 'confidence_drift',
                    'direction': 'down' if conf_change < 0 else 'up',
                    'change_pct': conf_change * 100,
                    'date': current_date
                })
        
        # Check latency drift
        if previous.get('avg_latency_ms', 0) > 0:
            lat_change = (current.get('avg_latency_ms', 0) - previous['avg_latency_ms']) / previous['avg_latency_ms']
            if lat_change > self.alert_threshold:  # Only alert on slowdown
                alerts.append({
                    'type': 'latency_drift',
                    'direction': 'up',
                    'change_pct': lat_change * 100,
                    'date': current_date
                })
        
        # Store alerts
        self.metrics['alerts'].extend(alerts)
        if alerts:
            self._save_metrics()
        
        return alerts
    
    def get_status(self) -> Dict:
        """
        Get current monitoring status.
        
        Returns:
            Dict with current status and key metrics
        """
        today = datetime.now().strftime('%Y-%m-%d')
        today_metrics = self.metrics['daily'].get(today, {})
        
        return {
            'monitoring_active': True,
            'log_path': self.log_path,
            'metrics_path': self.metrics_path,
            'today': today_metrics,
            'evaluation_targets': EVALUATION_TARGETS,
            'recent_alerts': self.metrics['alerts'][-5:] if self.metrics['alerts'] else [],
            'buffer_size': len(self.log_buffer)
        }
    
    def close(self):
        """Cleanup and save all pending data."""
        self._flush_buffer()
        self._save_metrics()


# Singleton instance
_monitor_instance = None

def get_monitor() -> ModelMonitor:
    """Get singleton monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MONITORING TEST")
    print("=" * 60)
    
    monitor = ModelMonitor(
        log_path="data/logs/test_predictions.jsonl",
        metrics_path="data/logs/test_metrics.json"
    )
    
    # Log some test predictions
    print("\n1. Logging test predictions...")
    for i in range(5):
        monitor.log_prediction(
            input_text=f"Test message {i} - hôm nay rất vui!",
            predicted_emotion="joy",
            predicted_intensity=0.8,
            suggested_emojis=["😊", "🎉", "✨"],
            confidence=0.75 + (i * 0.02),
            latency_ms=150 + (i * 10)
        )
    
    # Compute daily metrics
    print("\n2. Computing daily metrics...")
    metrics = monitor.compute_daily_metrics()
    print(f"   Total predictions: {metrics.total_predictions}")
    print(f"   Avg confidence: {metrics.avg_confidence:.2f}")
    print(f"   Avg latency: {metrics.avg_latency_ms:.1f}ms")
    
    # Get status
    print("\n3. Monitor status:")
    status = monitor.get_status()
    print(f"   Active: {status['monitoring_active']}")
    print(f"   Targets: {list(status['evaluation_targets'].keys())}")
    
    # Cleanup
    monitor.close()
    print("\n✓ Monitoring test complete!")

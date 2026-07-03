import math
from pathlib import Path

from comparison import DSMCentralizedComparison
from queue_models import QueueModel


ROOT = Path(__file__).resolve().parents[1]


def test_large_multiserver_queue_is_numerically_stable():
    queue = QueueModel(
        arrival_rate=0.006,
        service_time=45_000.0,
        servers=480,
        Ca2=1.0,
        Cs2=0.5,
    )

    delay = queue.waiting_time()

    assert math.isfinite(delay)
    assert delay >= 0.0


def test_thesis_strong_config_has_jointly_stable_latency_points():
    comparison = DSMCentralizedComparison(str(ROOT / "config_thesis_strong.yaml"))

    result = comparison.comprehensive_comparison()

    assert max(comparison.config["system"]["arrival_rates"]) == 0.006
    assert result.performance_advantage["latency_sample_count"] > 0
    assert math.isfinite(result.performance_advantage["avg_latency_improvement"])

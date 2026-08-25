from __future__ import annotations

import re
import threading
from collections import Counter, defaultdict


JOB_PATH = re.compile(r"^/v1/jobs/[^/]+(?:/events)?$")


def normalized_path(path: str) -> str:
    if JOB_PATH.match(path):
        return "/v1/jobs/{job_id}/events" if path.endswith("/events") else "/v1/jobs/{job_id}"
    return path


class HttpMetrics:
    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = Counter()
        self._durations = defaultdict(list)

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        key = (method, normalized_path(path), str(status))
        with self._lock:
            self._requests[key] += 1
            self._durations[key].append(seconds)

    @staticmethod
    def _labels(key) -> str:
        method, path, status = key
        return f'method="{method}",path="{path}",status="{status}"'

    def render(self, job_counts: dict[str, int]) -> str:
        lines = [
            "# HELP geomed_http_requests_total HTTP requests by method, normalized path, and status.",
            "# TYPE geomed_http_requests_total counter",
        ]
        with self._lock:
            for key, count in sorted(self._requests.items()):
                labels = self._labels(key)
                lines.append(f"geomed_http_requests_total{{{labels}}} {count}")
            lines += [
                "# HELP geomed_http_request_duration_seconds HTTP request latency.",
                "# TYPE geomed_http_request_duration_seconds histogram",
            ]
            for key, values in sorted(self._durations.items()):
                labels = self._labels(key)
                for bucket in self.BUCKETS:
                    count = sum(value <= bucket for value in values)
                    lines.append(f'geomed_http_request_duration_seconds_bucket{{{labels},le="{bucket}"}} {count}')
                lines.append(f'geomed_http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {len(values)}')
                lines.append(f"geomed_http_request_duration_seconds_sum{{{labels}}} {sum(values)}")
                lines.append(f"geomed_http_request_duration_seconds_count{{{labels}}} {len(values)}")
        lines += ["# HELP geomed_jobs Jobs by current status.", "# TYPE geomed_jobs gauge"]
        for status, count in sorted(job_counts.items()):
            lines.append(f'geomed_jobs{{status="{status}"}} {count}')
        return "\n".join(lines) + "\n"

"""metrics.py — Logfire metric definitions shared across the pipeline.

Metrics complement traces: a trace shows one `ask()` call end-to-end, but a
dashboard needs aggregates across many calls (volume, grounded rate, latency,
error rate) — that's what these counters/histograms are for.
"""

import logfire

queries_total = logfire.metric_counter(
    "queries_total", unit="1", description="Number of ask() calls, tagged by backend/embedding_model/collection_name."
)
answers_total = logfire.metric_counter(
    "answers_total", unit="1", description="Number of generated answers, tagged by backend/grounded."
)
truncated_total = logfire.metric_counter(
    "truncated_total", unit="1", description="Number of generations that hit the max_tokens cap, tagged by backend."
)
generation_duration_seconds = logfire.metric_histogram(
    "generation_duration_seconds", unit="s", description="Wall-clock generation time, tagged by backend."
)
errors_total = logfire.metric_counter(
    "errors_total", unit="1", description="Number of ask() calls that raised an exception, tagged by backend."
)

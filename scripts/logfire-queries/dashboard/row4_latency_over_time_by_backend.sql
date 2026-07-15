-- Row 4 time series:
-- Name: Latency over time by backend
-- Description: Average generation time in seconds per minute, one column per backend (Gemma vs. Claude) —
-- shows whether either backend drifts slower over a session, not just its overall average.
-- Type: Time Series
-- Note: backend is pivoted into separate columns (rather than a `backend` grouping row, as the
-- pie queries used) so this plots as two lines even if the panel has no "split by" support.
SELECT
  date_trunc('minute', start_timestamp) AS time,
  round(avg((attributes->>'duration_seconds')::numeric) FILTER (WHERE attributes->>'backend' = 'google/gemma-2-2b-it'), 2) AS gemma_avg_duration_seconds,
  round(avg((attributes->>'duration_seconds')::numeric) FILTER (WHERE attributes->>'backend' = 'claude-haiku-4-5-20251001'), 2) AS claude_avg_duration_seconds
FROM records
WHERE span_name IN ('local generation complete', 'api generation complete')
GROUP BY time
ORDER BY time

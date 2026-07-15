-- Row 4 time series:
-- Name: Grounded rate over time
-- Description: Percentage of answers grounded in the retrieved corpus per minute — shows whether
-- answer quality is stable, improving, or degrading over a session.
-- Type: Time Series
SELECT
  date_trunc('minute', start_timestamp) AS time,
  round(100.0 * count(*) FILTER (WHERE (attributes->>'grounded')::boolean) / count(*), 1) AS grounded_rate_pct
FROM records
WHERE span_name = 'answer parsed'
GROUP BY time
ORDER BY time

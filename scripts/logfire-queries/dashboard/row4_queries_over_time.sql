-- Row 4 time series:
-- Name: Queries over time
-- Description: Number of ask() calls per minute — usage/traffic trend.
-- Type: Time Series
SELECT
  date_trunc('minute', start_timestamp) AS time,
  count(*) AS queries
FROM records
WHERE span_name = 'ask'
GROUP BY time
ORDER BY time

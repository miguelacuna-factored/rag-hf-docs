-- Row 4 time series:
-- Name: Errors over time
-- Description: Number of ask() calls that raised an exception per minute — pairs with the error-rate alert.
-- Type: Time Series
SELECT
  date_trunc('minute', start_timestamp) AS time,
  count(*) FILTER (WHERE is_exception) AS errors
FROM records
WHERE span_name = 'ask'
GROUP BY time
ORDER BY time

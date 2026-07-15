-- Row 2 by-backend comparison:
-- Name: Error rate by backend
-- Description: Percentage of queries that raised an exception, split by backend.
-- Type: Bar Chart
SELECT
  attributes->>'backend' AS backend,
  count(*) FILTER (WHERE is_exception) AS errors,
  count(*) AS total_asks,
  round(100.0 * count(*) FILTER (WHERE is_exception) / count(*), 1) AS error_rate_pct
FROM records
WHERE span_name = 'ask'
GROUP BY backend

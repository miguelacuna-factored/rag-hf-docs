-- Row 1 big number:
-- Name: Error rate %
-- Description: Percentage of queries that raised an exception instead of returning an answer.
-- Type: Values
SELECT round(100.0 * count(*) FILTER (WHERE is_exception) / count(*), 1) AS error_rate_pct
FROM records
WHERE span_name = 'ask'

-- Row 1 big number:
-- Name: Total queries
-- Description: Number of ask() calls handled by the app — overall usage volume.
-- Type: Values
SELECT count(*) AS total_queries
FROM records
WHERE span_name = 'ask'

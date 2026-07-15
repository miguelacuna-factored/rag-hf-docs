-- Row 2 by-backend comparison:
-- Name: Queries by backend
-- Description: Number of ask() calls, split by backend (Gemma vs. Claude).
-- Type: Bar Chart
SELECT
  attributes->>'backend' AS backend,
  count(*) AS queries
FROM records
WHERE span_name = 'ask'
GROUP BY backend

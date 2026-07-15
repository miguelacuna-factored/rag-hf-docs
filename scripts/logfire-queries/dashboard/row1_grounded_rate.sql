-- Row 1 big number:
-- Name: Grounded rate %
-- Description: Percentage of answers actually cited from the retrieved corpus, vs. the model saying it doesn't know.
-- Type: Values
SELECT round(100.0 * count(*) FILTER (WHERE (attributes->>'grounded')::boolean) / count(*), 1) AS grounded_rate_pct
FROM records
WHERE span_name = 'answer parsed'

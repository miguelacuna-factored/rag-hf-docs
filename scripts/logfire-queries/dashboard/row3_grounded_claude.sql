-- Row 3 by-backend comparison:
-- Name: Grounded vs. ungrounded — Claude
-- Description: Within the Claude Haiku backend, the split between grounded and ungrounded answers.
-- Type: Pie Chart
SELECT
  CASE WHEN (attributes->>'grounded')::boolean THEN 'Grounded' ELSE 'Ungrounded' END AS label,
  count(*) AS value
FROM records
WHERE span_name = 'answer parsed'
  AND attributes->>'backend' = 'claude-haiku-4-5-20251001'
GROUP BY label

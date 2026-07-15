-- Row 3 by-backend comparison:
-- Name: Grounded vs. ungrounded — Gemma
-- Description: Within the local Gemma backend, the split between grounded and ungrounded answers.
-- Type: Pie Chart
SELECT
  CASE WHEN (attributes->>'grounded')::boolean THEN 'Grounded' ELSE 'Ungrounded' END AS label,
  count(*) AS value
FROM records
WHERE span_name = 'answer parsed'
  AND attributes->>'backend' = 'google/gemma-2-2b-it'
GROUP BY label

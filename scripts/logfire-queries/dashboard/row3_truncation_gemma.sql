-- Row 3 by-backend comparison:
-- Name: Truncated vs. complete — Gemma
-- Description: Within the local Gemma backend, the split between truncated and complete generations.
-- Truncated means the answer hit the max_tokens cap and got cut off before reaching a
-- natural stopping point, instead of ending because the model was actually done.
-- Type: Pie Chart
SELECT
  CASE WHEN (attributes->>'truncated')::boolean THEN 'Truncated' ELSE 'Complete' END AS label,
  count(*) AS value
FROM records
WHERE span_name = 'local generation complete'
  AND attributes->>'backend' = 'google/gemma-2-2b-it'
GROUP BY label

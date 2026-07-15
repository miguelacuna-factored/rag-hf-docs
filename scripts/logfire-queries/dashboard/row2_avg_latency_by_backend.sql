-- Row 2 by-backend comparison:
-- Name: Latency by backend
-- Description: Average generation time in seconds, split by backend (Gemma vs. Claude).
-- Type: Bar Chart
SELECT
  attributes->>'backend' AS backend,
  round(avg((attributes->>'duration_seconds')::numeric), 2) AS avg_duration_seconds,
  count(*) AS generations
FROM records
-- backend was only added to these events partway through the session; older rows
-- predate that fix and have no backend attribute, so exclude the resulting NULL bucket.
WHERE span_name IN ('local generation complete', 'api generation complete')
  AND attributes->>'backend' IS NOT NULL
GROUP BY backend

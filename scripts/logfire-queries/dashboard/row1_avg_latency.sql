-- Row 1 big number:
-- Name: Avg latency
-- Description: Average end-to-end generation time in seconds, across both backends.
-- Type: Values
SELECT round(avg((attributes->>'duration_seconds')::numeric), 2) AS avg_duration_seconds
FROM records
WHERE span_name IN ('local generation complete', 'api generation complete')

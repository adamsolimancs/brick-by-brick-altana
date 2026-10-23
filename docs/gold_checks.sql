-- Run after a successful gold update, against a consistent source snapshot.
SELECT 'clean_input' AS stage, COUNT(*) AS rows
FROM addressresolution.silver.cleaned_1_0
UNION ALL
SELECT 'gold_accounted_input', COALESCE(SUM(input_row_count), 0)
FROM addressresolution.gold.entities_1_0
UNION ALL
SELECT 'review_input', COUNT(*) FROM addressresolution.gold.input_review_1_0;
-- clean_input must equal gold_accounted_input + review_input.

SELECT entity_id, COUNT(*) AS copies
FROM addressresolution.gold.entities_1_0
GROUP BY entity_id HAVING entity_id IS NULL OR COUNT(*) <> 1;
-- Must return zero rows.

SELECT reason, COUNT(*) AS rows
FROM addressresolution.gold.input_review_1_0
LATERAL VIEW EXPLODE(gold_input_reasons) AS reason
GROUP BY reason ORDER BY rows DESC;

SELECT resolution_status, COUNT(*) AS entities, SUM(source_record_count) AS sources
FROM addressresolution.gold.entities_1_0
GROUP BY resolution_status;

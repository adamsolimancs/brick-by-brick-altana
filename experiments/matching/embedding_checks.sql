-- Replace :embedding_run_id with a COMPLETED embedding run from the first query.
SELECT embedding_run_id, source_run_id, status,
       get_json_object(manifest_json, '$.new_candidate_pairs') AS new_pairs,
       get_json_object(manifest_json, '$.exact_search_audit') AS geometric_recall_audit
FROM addressresolution.matching_experiments_aes10130.embedding_runs
ORDER BY get_json_object(manifest_json, '$.completed_at') DESC;

-- Each accepted source record should have exactly one row per view.
SELECT view, count(*) AS vector_rows, count(DISTINCT unique_id) AS unique_records,
       count_if(embedding IS NULL OR size(embedding) != 384
         OR exists(embedding, x -> x IS NULL OR isnan(x))
         OR abs(aggregate(embedding, CAST(0 AS DOUBLE), (acc, x) -> acc + x*x) - 1.0) > 0.0001
       ) AS invalid_vectors
FROM addressresolution.matching_experiments_aes10130.embedding_records
WHERE embedding_run_id = :embedding_run_id
GROUP BY view;

-- These counts must all be zero.
SELECT count_if(e.id_l >= e.id_r) AS invalid_pair_orientation,
       count_if(e.approved_for_merge) AS unintended_approved_merges,
       count_if(p.unique_id_l IS NOT NULL) AS old_pair_overlap,
       count_if(e.retrieval_score IS NULL OR isnan(e.retrieval_score)
         OR e.retrieval_score < -1 OR e.retrieval_score > 1) AS invalid_scores,
       count_if((e.country_conflict OR e.region_conflict OR e.house_conflict
         OR e.unit_conflict OR e.numeric_conflict)
         AND e.decision <> 'review_conflict') AS unflagged_conflicts
FROM addressresolution.matching_experiments_aes10130.embedding_candidates e
LEFT JOIN addressresolution.matching_experiments_aes10130.pair_scores p
  ON p.experiment_run_id = e.source_run_id
 AND p.unique_id_l = e.id_l AND p.unique_id_r = e.id_r
WHERE e.embedding_run_id = :embedding_run_id;

SELECT count(*) - count(DISTINCT struct(id_l,id_r)) AS duplicate_pairs
FROM addressresolution.matching_experiments_aes10130.embedding_candidates
WHERE embedding_run_id = :embedding_run_id;

SELECT decision, retrieval_views, count(*) AS pair_count,
       min(retrieval_score) AS min_cosine, max(retrieval_score) AS max_cosine
FROM addressresolution.matching_experiments_aes10130.embedding_candidates
WHERE embedding_run_id = :embedding_run_id
GROUP BY decision, retrieval_views;

SELECT *
FROM addressresolution.matching_experiments_aes10130.embedding_review_sample
WHERE embedding_run_id = :embedding_run_id
ORDER BY decision, retrieval_score DESC;

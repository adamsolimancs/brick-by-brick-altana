-- Read-only checks for the most recent COMPLETED experiment.
-- Failed/partial runs are deliberately excluded.
WITH latest AS (
  SELECT experiment_run_id FROM addressresolution.matching_experiments_aes10130.runs
  WHERE status = 'COMPLETED'
  ORDER BY get_json_object(manifest_json, '$.completed_at') DESC LIMIT 1
), scores AS (
  SELECT p.* FROM addressresolution.matching_experiments_aes10130.pair_scores p
  INNER JOIN latest USING (experiment_run_id)
)
SELECT
  COUNT(*) AS scored_pairs,
  COUNT(*) - COUNT(DISTINCT struct(unique_id_l, unique_id_r)) AS duplicate_pairs,
  COUNT_IF(unique_id_l >= unique_id_r) AS invalid_pair_order,
  COUNT_IF(match_probability IS NULL OR isnan(match_probability)
    OR match_probability < 0 OR match_probability > 1) AS invalid_probabilities,
  COUNT_IF(approved_for_merge) AS approved_merges,
  COUNT_IF(decision = 'proposed_match' AND (
    house_conflict OR unit_conflict OR numeric_conflict OR region_conflict
    OR unit_missing_one_side OR match_probability < 0.99
    OR name_similarity < 0.85 OR street_similarity < 0.85 OR city_similarity < 0.85
  )) AS unsafe_proposals,
  COUNT_IF(exact_baseline) AS retained_baseline_pairs
FROM scores;

-- Candidate-pass marginal contribution, in the implemented pass order.
WITH latest AS (
  SELECT experiment_run_id FROM addressresolution.matching_experiments_aes10130.runs
  WHERE status = 'COMPLETED'
  ORDER BY get_json_object(manifest_json, '$.completed_at') DESC LIMIT 1
), passes AS (
  SELECT p.*, CASE blocking_pass
    WHEN 'exact_baseline' THEN 1 WHEN 'region_city_house' THEN 2
    WHEN 'region_name_prefix' THEN 3 WHEN 'country_name_rescue' THEN 4
    WHEN 'city_road_prefix' THEN 5 END AS pass_order
  FROM addressresolution.matching_experiments_aes10130.candidate_passes p
  INNER JOIN latest USING (experiment_run_id)
), ranked AS (
  SELECT *, row_number() OVER (PARTITION BY id_l, id_r ORDER BY pass_order) AS contribution_rank
  FROM passes
)
SELECT blocking_pass, COUNT(*) AS raw_pairs,
  COUNT_IF(contribution_rank = 1) AS additional_unique_pairs
FROM ranked GROUP BY blocking_pass ORDER BY MIN(pass_order);

-- Compare outcomes on identical clean/translated inputs.
WITH latest AS (
  SELECT experiment_run_id FROM addressresolution.matching_experiments_aes10130.runs
  WHERE status = 'COMPLETED'
  ORDER BY get_json_object(manifest_json, '$.completed_at') DESC LIMIT 1
)
SELECT decision, input_route_l, input_route_r, COUNT(*) AS pair_count
FROM addressresolution.matching_experiments_aes10130.pair_scores
INNER JOIN latest USING (experiment_run_id)
GROUP BY decision, input_route_l, input_route_r
ORDER BY decision, input_route_l, input_route_r;

-- Reviewable hypotheses, not confirmed additional matches.
WITH latest AS (
  SELECT experiment_run_id FROM addressresolution.matching_experiments_aes10130.runs
  WHERE status = 'COMPLETED'
  ORDER BY get_json_object(manifest_json, '$.completed_at') DESC LIMIT 1
)
SELECT unique_id_l, unique_id_r, company_l, company_r,
  street_address_l, street_address_r, country_l, match_probability,
  probability_prior_2x, probability_prior_4x, blocking_passes
FROM addressresolution.matching_experiments_aes10130.pair_scores
INNER JOIN latest USING (experiment_run_id)
WHERE decision = 'proposed_match'
ORDER BY match_probability DESC, unique_id_l, unique_id_r LIMIT 30;

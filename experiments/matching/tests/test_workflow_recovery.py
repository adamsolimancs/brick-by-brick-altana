"""Recovery must consume successful upstream attempts, never partial outputs."""

import re
import unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def job(name):
    value = yaml.load(
        (ROOT / "resources" / f"{name}.job.yml").read_text(), Loader=UniqueLoader
    )
    return value["resources"]["jobs"][name]


class WorkflowRecoveryTests(unittest.TestCase):
    def test_repaired_tasks_get_new_ids_and_consume_completed_upstream_ids(self):
        config = job("address_deduplication")
        tasks = {t["task_key"]: t for t in config["tasks"]}
        for task_key, output, input_key, upstream, upstream_output in [
            (
                "embed_snapshot",
                "embedding_run_id",
                "source_run_id",
                "block_and_score",
                "experiment_run_id",
            ),
            (
                "search_hnsw",
                "embedding_run_id",
                "source_embedding_run_id",
                "embed_snapshot",
                "embedding_run_id",
            ),
            (
                "compare_strings",
                "string_scoring_run_id",
                "input_embedding_run_id",
                "search_hnsw",
                "embedding_run_id",
            ),
            (
                "unified_scoring",
                "unified_run_id",
                "input_string_scoring_run_id",
                "compare_strings",
                "string_scoring_run_id",
            ),
            (
                "resolve_addresses",
                "deduplication_run_id",
                "unified_run_id",
                "unified_scoring",
                "unified_run_id",
            ),
        ]:
            params = tasks[task_key]["notebook_task"]["base_parameters"]
            self.assertIn("{{task.execution_count}}", params[output])
            self.assertEqual(
                params[input_key],
                "{{tasks." + upstream + ".values." + upstream_output + "}}",
            )
            # Job parameters override task parameters: do not shadow attempt IDs.
            names = {p["name"] for p in config["parameters"]}
            self.assertNotIn(output, names)
            self.assertNotIn(input_key, names)
            upstream_file = tasks[upstream]["notebook_task"]["notebook_path"]
            source = (ROOT / "resources" / upstream_file).resolve().read_text()
            self.assertRegex(
                source,
                re.compile(
                    r'taskValues.set\(key="' + upstream_output + '", value=run_id\)'
                ),
            )

    def test_cache_checkpoint_runs_after_failure_but_scoring_requires_silver_success(
        self,
    ):
        tasks = {
            t["task_key"]: t for t in job("silver_to_address_deduplication")["tasks"]
        }
        self.assertEqual(
            tasks["refresh_silver"]["depends_on"],
            [{"task_key": "preserve_translation_cache"}],
        )
        self.assertEqual(tasks["checkpoint_translation_cache"]["run_if"], "ALL_DONE")
        self.assertEqual(tasks["score_and_deduplicate"]["run_if"], "ALL_SUCCESS")
        self.assertEqual(
            {d["task_key"] for d in tasks["score_and_deduplicate"]["depends_on"]},
            {"refresh_silver", "checkpoint_translation_cache"},
        )


if __name__ == "__main__":
    unittest.main()

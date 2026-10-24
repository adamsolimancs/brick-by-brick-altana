"""Run with python -m unittest discover -s tests (requires PySpark 3.5+).

Exercise the provider cleaning step on small in-memory bronze tables.
provider_cleaning.py imports address_cleaning_utils from a teammate's
workspace folder; fixtures/address_cleaning_utils.py is a byte-identical
copy of /Workspace/Users/jl13852@nyu.edu/address_cleaning_utils.py, served
under that module name. Re-export it when the workspace file changes.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import pyspark
    from pyspark.sql import DataFrameReader, SparkSession, functions as F
except ImportError as exc:
    raise unittest.SkipTest("PySpark is required for provider cleaning tests") from exc


TESTS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_decorator(*args, **kwargs):
    return lambda function: function


address_cleaning_utils = _load(
    "address_cleaning_utils", TESTS / "fixtures" / "address_cleaning_utils.py"
)
with (
    patch.dict(sys.modules, {"address_cleaning_utils": address_cleaning_utils}),
    patch.object(
        pyspark,
        "pipelines",
        SimpleNamespace(
            materialized_view=_identity_decorator,
            expect_all_or_fail=_identity_decorator,
        ),
        create=True,
    ),
):
    provider_cleaning = _load(
        "provider_cleaning",
        TESTS.parent / "silver_pipeline" / "transformations" / "provider_cleaning.py",
    )


PROVIDER_A_RAW = "addressresolution.bronze.provider_a_raw"
PROVIDER_B_RAW = "addressresolution.bronze.provider_b_raw"
# Bronze raw tables hold every source value as STRING.
PROVIDER_A_SCHEMA = (
    "provider_a_id STRING, company STRING, street_address STRING, city STRING, "
    "state STRING, country STRING, updated_at STRING"
)
PROVIDER_B_SCHEMA = (
    "provider_b_id STRING, name STRING, street_address STRING, locality STRING, "
    "admin_region STRING, country STRING, created_at STRING"
)


class ProviderCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Spark workers must use this interpreter, not the system python3.
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("provider-cleaning-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def clean(self, dataset, table, rows, schema):
        """Run a provider cleaning dataset on rows served as its bronze table."""
        frames = {table: self.spark.createDataFrame(rows, schema)}
        # The helper calls F.try_to_date, which PySpark 4 adds. Under Spark
        # 3.5's default non-ANSI mode, to_date also returns NULL for bad input.
        try_to_date = getattr(F, "try_to_date", F.to_date)
        with (
            patch.object(F, "try_to_date", try_to_date, create=True),
            patch.object(provider_cleaning, "spark", self.spark, create=True),
            patch.object(
                DataFrameReader, "table", side_effect=lambda name: frames[name]
            ),
        ):
            return {r.provider_id: r for r in dataset().collect()}

    def test_provider_a_full_address_uses_final_country_code(self):
        rows = self.clean(
            provider_cleaning.provider_a_clean,
            PROVIDER_A_RAW,
            [
                (
                    "1",
                    "Acme",
                    "10 Main Street",
                    "Springfield",
                    "Illinois",
                    "United States",
                    "2024-01-02",
                ),
                ("2", "Beta", "20 Oak Road", "Chicago", "Illinois", None, "2024-01-03"),
                ("3", "Gamma", "30 Elm Lane", "Leeds", "Atlantis", None, "2024-01-04"),
            ],
            PROVIDER_A_SCHEMA,
        )
        self.assertEqual(rows[1].country, "US")
        self.assertEqual(
            rows[1].full_address, "10 Main Street, Springfield, Illinois, US"
        )
        # Country inferred from state, then mapped to its code.
        self.assertTrue(rows[2].country_inferred)
        self.assertEqual(rows[2].full_address, "20 Oak Road, Chicago, Illinois, US")
        # No country to infer: full_address leaves it out, as concat_ws does.
        self.assertIsNone(rows[3].country)
        self.assertEqual(rows[3].full_address, "30 Elm Lane, Leeds, Atlantis")

    def test_provider_b_full_address_includes_inferred_country(self):
        rows = self.clean(
            provider_cleaning.provider_b_clean,
            PROVIDER_B_RAW,
            [
                ("1", "Acme", "10 Main Street", "Hartford", "CT", "US", "2024-01-02"),
                ("2", "Beta", "20 Oak Road", "New Haven", "CT", None, "2024-01-03"),
            ],
            PROVIDER_B_SCHEMA,
        )
        self.assertEqual(rows[1].full_address, "10 Main Street, Hartford, CT, US")
        self.assertTrue(rows[2].country_inferred)
        self.assertEqual(rows[2].full_address, "20 Oak Road, New Haven, CT, US")


if __name__ == "__main__":
    unittest.main()

"""
test_transform.py

Unit tests for the spatial transformations and database ingestion inside dags/dag_pipeline.py.
We use standard unittest mocks to simulate API calls and database connections.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import h3
import pandas as pd
from shapely.geometry import LineString, Polygon

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dags.dag_pipeline import (
    create_merged_polygon_from_hexes,
    get_speed_category,
    transform_line_to_point,
    transform_traffic_data,
)


class TestTransformTrafficData(unittest.TestCase):
    def test_get_speed_category(self):
        """Test classification of speeds into slow, medium, fast, and unknown."""
        self.assertEqual(get_speed_category(15), "Slow (0-20 km/h)")
        self.assertEqual(get_speed_category(20), "Slow (0-20 km/h)")
        self.assertEqual(get_speed_category(35), "Medium (20-50 km/h)")
        self.assertEqual(get_speed_category(50), "Medium (20-50 km/h)")
        self.assertEqual(get_speed_category(65), "Fast (>50 km/h)")
        self.assertEqual(get_speed_category(None), "Unknown")
        self.assertEqual(get_speed_category(float("nan")), "Unknown")

    def test_transform_line_to_point(self):
        """Test that a LineString (EPSG:2154) is correctly split into points."""
        # Create a 15-meter long line starting at (1840000, 5175000) going east
        line = LineString([(1840000, 5175000), (1840015, 5175000)])

        # Run interpolation
        points = transform_line_to_point(line)

        # 15 meters with 7-meter steps should yield exactly 4 points:
        # d=0 (start), d=7, d=14, d=15 (end point fallback)
        self.assertEqual(len(points), 4)

    def test_create_merged_polygon_from_hexes_empty(self):
        """Test H3 boundary merging with empty lists."""
        self.assertIsNone(create_merged_polygon_from_hexes([]))
        self.assertIsNone(create_merged_polygon_from_hexes(None))

    def test_create_merged_polygon_from_hexes_valid(self):
        """Test H3 boundary merging with sample H3 hex indices."""
        # Dynamically generate a valid Lyon cell ID to bypass strict H3 v4 cell validation
        valid_hex = h3.latlng_to_cell(45.75, 4.85, 13)
        hex_list = [valid_hex, valid_hex]  # Duplicates should be handled
        polygon = create_merged_polygon_from_hexes(hex_list)

        self.assertIsNotNone(polygon)
        self.assertTrue(isinstance(polygon, Polygon))

    @patch("dags.dag_pipeline.create_engine")
    @patch("dags.dag_pipeline.os.makedirs")
    @patch("dags.dag_pipeline.pd.DataFrame.to_csv")
    @patch("dags.dag_pipeline.GeoDataFrame.to_file")
    def test_transform_traffic_data_pipeline_and_silver_push(
        self, mock_to_file, mock_to_csv, mock_makedirs, mock_create_engine
    ):
        """Test the complete data transformation pipeline and database Silver layer push."""

        # Setup lists to capture the DataFrame passed as 'self' and kwargs to the mocked to_sql method
        pushed_dfs = []
        pushed_kwargs = []

        def custom_to_sql(df_instance, *args, **kwargs):
            pushed_dfs.append(df_instance)
            pushed_kwargs.append(kwargs)
            return None

        # Mock database return value for fetching bronze layer
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Mocking single-row return containing the raw Grand Lyon API JSON payload
        mock_result.fetchone.return_value = (
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "libelle": "Avenue Jean Jaurès",
                            "vitesse": "40 km/h",
                            "est_a_jour": True,
                            "sens": "Direct",
                            "etat": "Fluide",
                            "last_update": "2026-05-29T21:00:00Z",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[1840000, 5175000], [1840013, 5175000]],  # 13 meters long
                        },
                    }
                ],
            },
        )

        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        # Run the transformation pipeline with our custom to_sql patch
        with patch("dags.dag_pipeline.pd.DataFrame.to_sql", new=custom_to_sql):
            transform_traffic_data()

        # Assertions for General flow
        mock_create_engine.assert_called_once()

        # In the consolidated pipeline, there are 2 SQL query executes:
        # 1. Fetching bronze data
        # 2. Executing 'CREATE SCHEMA IF NOT EXISTS silver;'
        self.assertEqual(mock_conn.execute.call_count, 2)

        mock_makedirs.assert_called_once_with("/opt/airflow/data", exist_ok=True)

        # Assertions for file backup storage exports
        mock_to_csv.assert_called_once()
        mock_to_file.assert_called_once()

        # Assertions for PostgreSQL Silver layer push
        self.assertEqual(len(pushed_dfs), 1)
        self.assertEqual(len(pushed_kwargs), 1)

        # Inspect arguments passed to pandas .to_sql()
        called_kwargs = pushed_kwargs[0]
        self.assertEqual(called_kwargs.get("name"), "trafic_vitesse_propre")
        self.assertEqual(called_kwargs.get("schema"), "silver")
        self.assertEqual(called_kwargs.get("if_exists"), "append")
        self.assertEqual(called_kwargs.get("index"), False)

        # Verify that the custom wrapper successfully captured the DataFrame
        df_result = pushed_dfs[0]
        self.assertTrue(isinstance(df_result, pd.DataFrame))

        # Verify that serialized geometry/list/dict columns exist
        self.assertIn("geometry_wgs84_wkt", df_result.columns)
        self.assertIn("points_json", df_result.columns)
        self.assertIn("hexes_json", df_result.columns)
        self.assertIn("merged_h3_geometry_json", df_result.columns)
        self.assertIn("transformed_at", df_result.columns)

        # Check that geometry coordinate WKT represents a 13-meter LineString
        self.assertTrue(df_result["geometry_wgs84_wkt"].iloc[0].startswith("LINESTRING"))

        # Check that list/dict fields are correctly dumped as JSON strings
        self.assertTrue(isinstance(df_result["points_json"].iloc[0], str))
        self.assertTrue(isinstance(df_result["hexes_json"].iloc[0], str))

        # Parse points and verify coordinate arrays
        parsed_points = json.loads(df_result["points_json"].iloc[0])
        self.assertEqual(
            len(parsed_points), 3
        )  # 13 meters with 7m interpolation interval -> d=0, d=7, d=13 (end point fallback)


if __name__ == "__main__":
    unittest.main()

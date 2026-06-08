"""
test_migrate_historical.py

Unit tests for the historical data migration module in utils/migrate_historical_to_silver.py.
We use standard unittest mocks to simulate file systems, GeoJSON read/write operations, and database connections.
"""

import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytz
from shapely.geometry import LineString

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.old_sans_interet.migrate_historical_to_silver import migrate_historical


class TestMigrateHistorical(unittest.TestCase):
    @patch("utils.old_sans_interet.migrate_historical_to_silver.create_engine")
    @patch("utils.old_sans_interet.migrate_historical_to_silver.glob.glob")
    @patch("utils.old_sans_interet.migrate_historical_to_silver.gpd.read_file")
    def test_migrate_historical_success(self, mock_read_file, mock_glob, mock_create_engine):
        """Cas nominal : 1 GeoJSON transformé, DB mockée, ingestion Silver.

        Vérifie :
          - `glob.glob` et `read_file` appelés une fois.
          - exactement 1 `mock_conn.execute` (CREATE SCHEMA).
          - `to_sql` appelé une fois avec les bons kwargs.
          - présence des colonnes sérialisées (`geometry_wgs84_wkt`,
            `points_json`, `hexes_json`, `merged_h3_geometry_json`,
            `transformed_at`).
          - `transformed_at` parsé depuis le nom de fichier en heure de Paris.
        """

        # 1. Mock file finder to return one test JSON file
        mock_glob.return_value = ["/opt/airflow/data/2026_05_29_21_00_transformed.json"]

        # 2. Mock reading of GeoJSON file
        # We create a simple GeoDataFrame with a LineString geometry and nested structures
        data = {
            "properties_libelle": ["Avenue Jean Jaurès"],
            "properties_sens": ["Direct"],
            "properties_etat": ["Fluide"],
            "properties_vitesse": [40.0],
            "properties_last_update": ["2026-05-29T21:00:00Z"],
            "properties_est_a_jour": [True],
            "speed_category": ["Medium (20-50 km/h)"],
            "speed_color_map": ["orange"],
            "points": [[[-4.85, 45.75], [-4.86, 45.76]]],
            "hexes": [["8d1f1a141b2e1ff"]],
            "merged_h3_geometry": [
                {"type": "Polygon", "coordinates": [[[-4.85, 45.75], [-4.86, 45.75], [-4.86, 45.76], [-4.85, 45.75]]]}
            ],
            "geometry": [LineString([(1840000, 5175000), (1840013, 5175000)])],
        }
        mock_gdf = gpd.GeoDataFrame(data, geometry="geometry")
        mock_read_file.return_value = mock_gdf

        # 3. Mock database engine
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        # Setup lists to capture values sent to to_sql
        pushed_dfs = []
        pushed_kwargs = []

        def custom_to_sql(df_instance, *args, **kwargs):
            pushed_dfs.append(df_instance)
            pushed_kwargs.append(kwargs)
            return None

        # Run historical migration
        with patch("utils.old_sans_interet.migrate_historical_to_silver.pd.DataFrame.to_sql", new=custom_to_sql):
            migrate_historical()

        # Assertions
        mock_create_engine.assert_called_once()
        mock_glob.assert_called_once()
        mock_read_file.assert_called_once_with("/opt/airflow/data/2026_05_29_21_00_transformed.json")

        # Verify schema checking
        mock_conn.execute.assert_called_once()

        # Verify database ingestion metrics
        self.assertEqual(len(pushed_dfs), 1)
        self.assertEqual(len(pushed_kwargs), 1)

        called_kwargs = pushed_kwargs[0]
        self.assertEqual(called_kwargs.get("name"), "trafic_vitesse_propre")
        self.assertEqual(called_kwargs.get("schema"), "silver")
        self.assertEqual(called_kwargs.get("if_exists"), "append")
        self.assertEqual(called_kwargs.get("index"), False)

        # Verify DataFrame structures and conversions
        df_result = pushed_dfs[0]
        self.assertTrue(isinstance(df_result, pd.DataFrame))

        # Check column mappings
        self.assertIn("geometry_wgs84_wkt", df_result.columns)
        self.assertIn("points_json", df_result.columns)
        self.assertIn("hexes_json", df_result.columns)
        self.assertIn("merged_h3_geometry_json", df_result.columns)
        self.assertIn("transformed_at", df_result.columns)

        # Check that spatial column represents a clean WKT
        self.assertTrue(df_result["geometry_wgs84_wkt"].iloc[0].startswith("LINESTRING"))

        # Check nested structures serialization to string
        self.assertTrue(isinstance(df_result["points_json"].iloc[0], str))
        self.assertTrue(isinstance(df_result["hexes_json"].iloc[0], str))
        self.assertTrue(isinstance(df_result["merged_h3_geometry_json"].iloc[0], str))

        # Check timestamp parsing
        expected_dt = pytz.timezone("Europe/Paris").localize(datetime(2026, 5, 29, 21, 0))
        self.assertEqual(df_result["transformed_at"].iloc[0], expected_dt)

    @patch("utils.old_sans_interet.migrate_historical_to_silver.create_engine")
    @patch("utils.old_sans_interet.migrate_historical_to_silver.glob.glob")
    def test_migrate_historical_empty_folder(self, mock_glob, mock_create_engine):
        """Cas dégradé : aucun fichier `*_transformed.json` dans le dossier.

        Le script doit s'arrêter proprement (warning loggué, pas d'ingestion).
        """
        mock_glob.return_value = []

        # Run historical migration
        migrate_historical()

        # Database check should not proceed further than engine creation & schema check
        mock_create_engine.assert_called_once()


if __name__ == "__main__":
    unittest.main()

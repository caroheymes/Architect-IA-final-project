"""
test_ingest.py

Unit tests for the ingestion module inside dags/dag_pipeline.py.
We use standard unittest mocks to simulate API calls and database connections.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dags.dag_pipeline import ingest_traffic_data


class TestIngestTrafficData(unittest.TestCase):
    @patch("dags.dag_pipeline.requests.get")
    @patch("dags.dag_pipeline.create_engine")
    def test_ingest_traffic_data_success(self, mock_create_engine, mock_get):
        """Test successful ingestion when API and Database are fully working."""

        # Mock API Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"libelle": "Rue de la République", "vitesse": 35, "est_a_jour": True},
                    "geometry": {"type": "LineString", "coordinates": [[1840000, 5175000], [1840100, 5175100]]},
                }
            ],
        }
        mock_get.return_value = mock_response

        # Mock Database Connection
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        # Run ingestion
        ingest_traffic_data()

        # Assertions
        mock_get.assert_called_once()
        mock_create_engine.assert_called_once()

        # Verify database begin was called
        mock_engine.begin.assert_called_once()

        # Verify execution of CREATE SCHEMA, CREATE TABLE, and INSERT queries
        self.assertTrue(mock_connection.execute.call_count >= 3)

    @patch("dags.dag_pipeline.requests.get")
    def test_ingest_traffic_data_api_failure(self, mock_get):
        """Test ingestion behavior when Grand Lyon API returns an error."""

        # Mock API Failure using requests.exceptions.RequestException
        mock_get.side_effect = requests.exceptions.RequestException("API connection timed out")

        # Run ingestion and verify that it raises an Exception
        with self.assertRaises(Exception) as context:
            ingest_traffic_data()

        self.assertIn("Ingestion failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()

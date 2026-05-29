# -*- coding: utf-8 -*-
"""
test_ingest.py

Unit tests for the 'ingest_api.py' script.
We use mocks to simulate API calls and database connections.
"""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingest_api import ingest_traffic_data

@patch('ingest_api.requests.get')
@patch('ingest_api.create_engine')
def test_ingest_traffic_data_success(mock_create_engine, mock_get):
    """Test successful ingestion when API and Database are fully working."""
    
    # Mock API Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "libelle": "Rue de la République",
                    "vitesse": 35,
                    "est_a_jour": True
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[1840000, 5175000], [1840100, 5175100]]
                }
            }
        ]
    }
    mock_get.return_value = mock_response

    # Mock Database Connection
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_connection
    mock_create_engine.return_value = mock_engine

    # Run ingestion
    result = ingest_traffic_data()

    # Assertions
    assert result is True
    mock_get.assert_called_once()
    mock_create_engine.assert_called_once()
    
    # Verify database begin was called
    mock_engine.begin.assert_called_once()
    
    # Verify execution of CREATE SCHEMA, CREATE TABLE, and INSERT queries
    assert mock_connection.execute.call_count >= 3

@patch('ingest_api.requests.get')
def test_ingest_traffic_data_api_failure(mock_get):
    """Test ingestion behavior when Grand Lyon API returns an error."""
    
    # Mock API Failure
    mock_get.side_effect = Exception("API connection timed out")

    # Run ingestion
    result = ingest_traffic_data()

    # Assertions
    assert result is False

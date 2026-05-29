# -*- coding: utf-8 -*-
"""
test_transform.py

Unit tests for the spatial transformations in 'transform_data.py'.
"""

import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest
from shapely.geometry import LineString, Polygon

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transform_data import (
    get_speed_category,
    transform_line_to_point,
    create_merged_polygon_from_hexes,
    transform_traffic_data
)

def test_get_speed_category():
    """Test classification of speeds into slow, medium, fast, and unknown."""
    assert get_speed_category(15) == "Slow (0-20 km/h)"
    assert get_speed_category(20) == "Slow (0-20 km/h)"
    assert get_speed_category(35) == "Medium (20-50 km/h)"
    assert get_speed_category(50) == "Medium (20-50 km/h)"
    assert get_speed_category(65) == "Fast (>50 km/h)"
    assert get_speed_category(None) == "Unknown"
    assert get_speed_category(float('nan')) == "Unknown"

def test_transform_line_to_point():
    """Test that a LineString (EPSG:2154) is correctly split into points."""
    # Create a 21-meter long line starting at (1840000, 5175000) going east
    line = LineString([(1840000, 5175000), (1840021, 5175000)])
    
    # Run interpolation
    points = transform_line_to_point(line)
    
    # 21 meters with 7-meter steps should yield exactly 4 points: 
    # d=0 (start), d=7, d=14, d=21 (end)
    assert len(points) == 4

def test_create_merged_polygon_from_hexes_empty():
    """Test H3 boundary merging with empty lists."""
    assert create_merged_polygon_from_hexes([]) is None
    assert create_merged_polygon_from_hexes(None) is None

def test_create_merged_polygon_from_hexes_valid():
    """Test H3 boundary merging with sample H3 hex indices."""
    # Using real H3 cell indices for Lyon area
    hex_list = ["8d1f1a141b2e1ff", "8d1f1a141b2e1ff"] # Duplicates should be handled
    polygon = create_merged_polygon_from_hexes(hex_list)
    
    assert polygon is not None
    assert isinstance(polygon, Polygon)

@patch('transform_data.create_engine')
@patch('transform_data.os.makedirs')
@patch('transform_data.pd.DataFrame.to_csv')
@patch('transform_data.GeoDataFrame.to_file')
def test_transform_traffic_data_pipeline(mock_to_file, mock_to_csv, mock_makedirs, mock_create_engine):
    """Test the complete data transformation pipeline end-to-end."""
    
    # Mock database return value
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ({
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
                    "last_update": "2026-05-29T21:00:00Z"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[1840000, 5175000], [1840014, 5175000]] # 14 meters long
                }
            }
        ]
    },)
    mock_conn.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_engine

    # Run the transformation pipeline
    success = transform_traffic_data()

    # Assertions
    assert success is True
    mock_create_engine.assert_called_once()
    mock_conn.execute.assert_called_once()
    mock_makedirs.assert_called_once_with("/opt/airflow/data", exist_ok=True)
    mock_to_csv.assert_called_once()
    mock_to_file.assert_called_once()
command_to_run: pytest -v

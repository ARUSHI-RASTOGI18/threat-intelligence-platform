"""
Automated Test Suite for Dynamic Data Layer & Offline Fallback
Location: ./tests/test_dynamic_pipeline.py
Execution: pytest tests/test_dynamic_pipeline.py
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import generate_demo_dataset
from src.dynamic_data import (
    normalize_dynamic_dataframe,
    compute_dynamic_country_signal,
    get_dynamic_feed_status,
    NORMALIZED_COLUMNS
)

def test_dynamic_schema_normalization():
    raw_mock = pd.DataFrame([{
        "event_id_cnty": "MOCK-001",
        "event_date": "2024-05-12",
        "country": "Iraq",
        "region": "Middle East",
        "latitude": 33.3152,
        "longitude": 44.3661,
        "event_type": "Explosions/Remote violence",
        "sub_event_type": "Air/drone strike",
        "actor1": "Unidentified Armed Group",
        "fatalities": 4,
        "notes": "Sample dynamic event description"
    }])

    norm_df = normalize_dynamic_dataframe(raw_mock)
    assert not norm_df.empty
    assert list(norm_df.columns) == NORMALIZED_COLUMNS
    assert norm_df.iloc[0]["country"] == "Iraq"
    assert norm_df.iloc[0]["data_source"] == "ACLED"
    assert norm_df.iloc[0]["year"] == 2024

def test_dynamic_empty_dataframe_handling():
    empty_norm = normalize_dynamic_dataframe(pd.DataFrame())
    assert empty_norm.empty
    assert list(empty_norm.columns) == NORMALIZED_COLUMNS

def test_dynamic_country_signal_calculation():
    gtd_df = generate_demo_dataset()
    mock_dyn = pd.DataFrame([
        {
            "event_id": "D-1", "event_date": "2024-01-01", "year": 2024, "month": 1,
            "country": "Iraq", "region": "Middle East", "latitude": 33.3, "longitude": 44.3,
            "event_type": "Explosions", "sub_event_type": "Shelling", "actor": "Group A",
            "fatalities": 2, "injured": 0, "notes": "", "data_source": "ACLED"
        }
    ])

    sig = compute_dynamic_country_signal(gtd_df, mock_dyn, "Iraq")
    assert "signal_tier" in sig
    assert "percentage_deviation" in sig
    assert "historical_annual_avg" in sig
    assert sig["country"] == "Iraq"

def test_dynamic_status_inspection_resilience():
    status = get_dynamic_feed_status()
    assert "status" in status
    assert "has_credentials" in status
    assert "cache_exists" in status

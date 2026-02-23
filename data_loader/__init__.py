"""
Data Loader Package

Provides functionality to load clinical trial data from various sources:
- ClinicalTrials.gov API
- Local JSON files
- Version history for dynamic CTKG
"""

from .ctgov_api import (
    fetch_trial_by_nct_id,
    fetch_trials_batch,
    fetch_version_history,
    fetch_all_versions_full,
    fetch_trials_with_amendments,
    CTGovAPIClient,
    get_client,
)
from .file_loader import load_trial_from_json, load_trials_from_json

__all__ = [
    'fetch_trial_by_nct_id',
    'fetch_trials_batch',
    'fetch_version_history',
    'fetch_all_versions_full',
    'fetch_trials_with_amendments',
    'CTGovAPIClient',
    'get_client',
    'load_trial_from_json',
    'load_trials_from_json'
]


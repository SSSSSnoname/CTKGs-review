"""
Version History Downloader using R cthist package

This module provides a Python wrapper for downloading clinical trial version
history using the R cthist package, which provides reliable access to the
ClinicalTrials.gov historical versions API.
"""

import subprocess
import json
import tempfile
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the R script for downloading version history
R_SCRIPT_PATH = Path(__file__).parent / "download_version_history.R"


def check_r_available() -> bool:
    """Check if R and cthist package are available."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(cthist); cat('OK')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return "OK" in result.stdout
    except Exception as e:
        logger.warning(f"R/cthist not available: {e}")
        return False


def download_version_history_r(nct_id: str, output_path: Optional[str] = None) -> Dict:
    """
    Download version history for a clinical trial using R cthist package.
    
    Args:
        nct_id: NCT identifier (e.g., "NCT02119676")
        output_path: Optional path to save the JSON output
        
    Returns:
        Dictionary containing version history data
    """
    # Use temp file if no output path specified
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        cleanup = True
    else:
        cleanup = False
    
    try:
        # Run R script
        cmd = ["Rscript", str(R_SCRIPT_PATH), nct_id, output_path]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        if result.returncode != 0:
            logger.error(f"R script failed: {result.stderr}")
            return {
                "success": False,
                "nct_id": nct_id,
                "message": f"R script error: {result.stderr}",
                "versions": []
            }
        
        # Read the output JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        return data
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading version history for {nct_id}")
        return {
            "success": False,
            "nct_id": nct_id,
            "message": "Timeout",
            "versions": []
        }
    except Exception as e:
        logger.error(f"Error downloading version history: {e}")
        return {
            "success": False,
            "nct_id": nct_id,
            "message": str(e),
            "versions": []
        }
    finally:
        if cleanup and os.path.exists(output_path):
            os.remove(output_path)


def convert_cthist_to_ctkg_format(cthist_data: Dict) -> List[Dict]:
    """
    Convert cthist output format to CTKG version format.
    
    cthist fields:
    - nctid, version_number, version_date
    - overall_status, whystopped
    - study_start_date, primary_completion_date
    - enrolment, enrolment_type
    - criteria (eligibility)
    - min_age, max_age, sex
    - outcome_measures
    - lead_sponsor, collaborators
    - locations, overall_contacts, central_contacts
    - references, responsible_party
    
    Args:
        cthist_data: Output from download_version_history_r
        
    Returns:
        List of version dictionaries in CTKG format
    """
    if not cthist_data.get("success", False):
        return []
    
    versions = cthist_data.get("versions", [])
    if not versions:
        return []
    
    ctkg_versions = []
    
    for i, version in enumerate(versions):
        # Map cthist fields to CTKG format
        ctkg_version = {
            "protocolSection": {
                "statusModule": {
                    "overallStatus": version.get("overall_status"),
                    "whyStopped": version.get("whystopped"),
                    "startDateStruct": {
                        "date": version.get("study_start_date")
                    },
                    "completionDateStruct": {
                        "date": version.get("primary_completion_date")
                    },
                    "primaryCompletionDateStruct": {
                        "date": version.get("primary_completion_date")
                    }
                },
                "designModule": {
                    "enrollmentInfo": {
                        "count": _safe_int(version.get("enrolment")),
                        "type": version.get("enrolment_type")
                    },
                    "phases": _parse_phases(version.get("phase")),
                    "studyType": version.get("study_type")
                },
                "conditionsModule": {
                    "conditions": _parse_list(version.get("conditions"))
                },
                "armsInterventionsModule": {
                    "interventions": _parse_json_field(version.get("interventions")),
                    "armGroups": []
                },
                "eligibilityModule": {
                    "eligibilityCriteria": version.get("criteria"),
                    "minimumAge": version.get("min_age"),
                    "maximumAge": version.get("max_age"),
                    "sex": version.get("sex")
                },
                "outcomesModule": {
                    "primaryOutcomes": _parse_outcomes(version.get("outcome_measures")),
                    "secondaryOutcomes": []
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": version.get("lead_sponsor")
                    },
                    "collaborators": _parse_list(version.get("collaborators"))
                },
                "contactsLocationsModule": {
                    "locations": _parse_json_field(version.get("locations")),
                    "centralContacts": _parse_json_field(version.get("central_contacts")),
                    "overallContacts": _parse_json_field(version.get("overall_contacts"))
                }
            },
            "_version_metadata": {
                "version_number": version.get("version_number", i + 1),
                "version_date": version.get("version_date"),
                "nct_id": version.get("nctid"),
                "total_versions": version.get("total_versions"),
                "results_posted": version.get("results_posted", False)
            },
            # Keep raw cthist data for direct access
            "_cthist_raw": version
        }
        
        ctkg_versions.append(ctkg_version)
    
    return ctkg_versions


def _parse_json_field(value) -> List:
    """Parse a JSON string field (used by cthist for complex fields)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            # Not JSON, try splitting
            if '|' in value:
                return [v.strip() for v in value.split('|') if v.strip()]
            return [value] if value else []
    return []




def _safe_int(value) -> Optional[int]:
    """Safely convert to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_phases(phase_str: str) -> List[str]:
    """Parse phase string into list."""
    if not phase_str:
        return []
    return [phase_str]


def _parse_list(value) -> List:
    """Parse a value that might be a list or string."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Try to split by common delimiters
        if '|' in value:
            return [v.strip() for v in value.split('|') if v.strip()]
        if ';' in value:
            return [v.strip() for v in value.split(';') if v.strip()]
        return [value]
    return []


def _parse_interventions(value) -> List[Dict]:
    """Parse intervention names into list of dicts."""
    items = _parse_list(value)
    return [{"name": item, "type": "Other"} for item in items]


def _parse_outcomes(value) -> List[Dict]:
    """Parse outcome measures into list of dicts."""
    items = _parse_list(value)
    return [{"measure": item} for item in items]


def get_version_history(nct_id: str, use_r: bool = True) -> List[Dict]:
    """
    Get version history for a clinical trial.
    
    Tries to use R cthist package first, then falls back to CT.gov API.
    
    Args:
        nct_id: NCT identifier
        use_r: Whether to attempt using R cthist package
        
    Returns:
        List of version dictionaries in CTKG format
    """
    if use_r and check_r_available():
        logger.info(f"Using R cthist to download version history for {nct_id}")
        cthist_data = download_version_history_r(nct_id)
        
        if cthist_data.get("success", False) and cthist_data.get("versions"):
            versions = convert_cthist_to_ctkg_format(cthist_data)
            if versions:
                logger.info(f"Downloaded {len(versions)} versions for {nct_id} via R cthist")
                return versions
    
    # Fallback to CT.gov API
    logger.info(f"Falling back to CT.gov API for {nct_id}")
    try:
        from .ctgov_api import get_client
        client = get_client()
        return client.get_all_study_versions_full(nct_id)
    except Exception as e:
        logger.error(f"Failed to get version history: {e}")
        return []


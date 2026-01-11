"""
ClinicalTrials.gov API Client

Downloads clinical trial data from the ClinicalTrials.gov v2 API.

API Documentation: https://clinicaltrials.gov/data-api/api
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# ClinicalTrials.gov API v2 base URL
BASE_URL = "https://clinicaltrials.gov/api/v2/"


class CTGovAPIClient:
    """
    Client for the ClinicalTrials.gov v2 API.
    
    This client provides methods to fetch trial data by NCT ID,
    search for trials, and download trial results.
    """
    
    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Dict:
        """
        Make an API request with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            
        Returns:
            JSON response as dictionary
        """
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
                    
        return {}
    
    def get_study(self, nct_id: str, fields: Optional[List[str]] = None) -> Dict:
        """
        Get a single study by NCT ID.
        
        Args:
            nct_id: NCT identifier (e.g., "NCT00256997")
            fields: Optional list of fields to return
            
        Returns:
            Study data as dictionary
        """
        endpoint = f"studies/{nct_id}"
        params = {}
        
        if fields:
            params['fields'] = ','.join(fields)
        
        return self._make_request(endpoint, params)
    
    def get_studies(
        self,
        nct_ids: List[str],
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get multiple studies by NCT IDs.
        
        Args:
            nct_ids: List of NCT identifiers
            fields: Optional list of fields to return
            
        Returns:
            List of study data dictionaries
        """
        results = []
        
        for nct_id in nct_ids:
            try:
                study = self.get_study(nct_id, fields)
                results.append(study)
            except Exception as e:
                logger.error(f"Failed to fetch {nct_id}: {e}")
                
        return results
    
    def search_studies(
        self,
        query: str,
        filter_conditions: Optional[Dict] = None,
        page_size: int = 100,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for studies using query parameters.
        
        Args:
            query: Search query string
            filter_conditions: Filter conditions (e.g., {"status": "COMPLETED"})
            page_size: Number of results per page
            max_results: Maximum total results to return
            
        Returns:
            List of matching studies
        """
        endpoint = "studies"
        params = {
            "query.term": query,
            "pageSize": page_size
        }
        
        if filter_conditions:
            for key, value in filter_conditions.items():
                params[f"filter.{key}"] = value
        
        results = []
        next_page_token = None
        
        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            
            response = self._make_request(endpoint, params)
            studies = response.get("studies", [])
            results.extend(studies)
            
            # Check if we've reached max results
            if max_results and len(results) >= max_results:
                results = results[:max_results]
                break
            
            # Check for next page
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        
        return results
    
    def get_study_versions(self, nct_id: str) -> List[Dict]:
        """
        Get version history for a study.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of version records sorted by date (oldest first)
        """
        # Note: This endpoint may vary based on API version
        endpoint = f"studies/{nct_id}/history"
        
        try:
            response = self._make_request(endpoint)
            versions = response.get("versions", [])
            # Sort by version date
            versions.sort(key=lambda v: v.get('versionDate', v.get('date', '')))
            return versions
        except Exception as e:
            logger.warning(f"Failed to get version history for {nct_id}: {e}")
            return []
    
    def get_study_version(self, nct_id: str, version: Union[int, str]) -> Dict:
        """
        Get a specific version of a study.
        
        Args:
            nct_id: NCT identifier
            version: Version number or version date string
            
        Returns:
            Study data for the specific version
        """
        endpoint = f"studies/{nct_id}"
        params = {}
        
        if isinstance(version, int):
            params['version'] = str(version)
        else:
            params['versionDate'] = version
        
        try:
            return self._make_request(endpoint, params)
        except Exception as e:
            logger.warning(f"Failed to get version {version} for {nct_id}: {e}")
            return {}
    
    def get_all_study_versions_full(
        self, 
        nct_id: str, 
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get full data for all versions of a study.
        
        This fetches the complete study data for each version,
        enabling detailed field-level comparison.
        
        Args:
            nct_id: NCT identifier
            fields: Optional list of fields to return per version
            
        Returns:
            List of full version records with complete study data
        """
        # First get the version list
        version_list = self.get_study_versions(nct_id)
        
        if not version_list:
            logger.info(f"No version history found for {nct_id}")
            return []
        
        full_versions = []
        
        for i, version_info in enumerate(version_list):
            version_date = version_info.get('versionDate', version_info.get('date'))
            
            if version_date:
                try:
                    full_data = self.get_study_version(nct_id, version_date)
                    if full_data:
                        full_data['_version_metadata'] = {
                            'version_number': i + 1,
                            'version_date': version_date,
                            'version_info': version_info
                        }
                        full_versions.append(full_data)
                except Exception as e:
                    logger.warning(f"Failed to fetch version {i+1} for {nct_id}: {e}")
        
        return full_versions
    
    def get_studies_with_version_history(
        self,
        query: str = "",
        min_versions: int = 2,
        max_results: int = 100
    ) -> List[str]:
        """
        Search for studies that have multiple versions (protocol amendments).
        
        Args:
            query: Optional search query
            min_versions: Minimum number of versions required
            max_results: Maximum number of NCT IDs to return
            
        Returns:
            List of NCT IDs with version history
        """
        # Search for studies
        if query:
            studies = self.search_studies(query, max_results=max_results * 2)
        else:
            # Default to completed studies with results
            studies = self.search_studies(
                "AREA[HasResults]true",
                max_results=max_results * 2
            )
        
        nct_ids_with_versions = []
        
        for study in studies:
            try:
                nct_id = study.get('protocolSection', {}).get(
                    'identificationModule', {}
                ).get('nctId', '')
                
                if nct_id:
                    versions = self.get_study_versions(nct_id)
                    if len(versions) >= min_versions:
                        nct_ids_with_versions.append(nct_id)
                        logger.info(f"Found {nct_id} with {len(versions)} versions")
                        
                        if len(nct_ids_with_versions) >= max_results:
                            break
            except Exception as e:
                logger.warning(f"Error checking versions: {e}")
                continue
        
        return nct_ids_with_versions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global client instance
_client = None

def get_client() -> CTGovAPIClient:
    """Get or create the global API client."""
    global _client
    if _client is None:
        _client = CTGovAPIClient()
    return _client


def fetch_trial_by_nct_id(nct_id: str) -> Dict:
    """
    Fetch a single trial by NCT ID.
    
    Args:
        nct_id: NCT identifier (e.g., "NCT00256997")
        
    Returns:
        Trial data as dictionary
        
    Example:
        >>> data = fetch_trial_by_nct_id("NCT00256997")
        >>> print(data['protocolSection']['identificationModule']['nctId'])
        NCT00256997
    """
    client = get_client()
    return client.get_study(nct_id)


def fetch_trials_batch(nct_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch multiple trials by NCT IDs.
    
    Args:
        nct_ids: List of NCT identifiers
        
    Returns:
        Dictionary mapping NCT IDs to trial data
        
    Example:
        >>> data = fetch_trials_batch(["NCT00256997", "NCT00123456"])
        >>> for nct_id, trial in data.items():
        ...     print(nct_id)
    """
    client = get_client()
    
    results = {}
    for nct_id in nct_ids:
        try:
            results[nct_id] = client.get_study(nct_id)
            logger.info(f"Fetched {nct_id}")
        except Exception as e:
            logger.error(f"Failed to fetch {nct_id}: {e}")
    
    return results


def fetch_trials_with_results(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    max_results: int = 1000
) -> List[Dict]:
    """
    Fetch trials that have posted results.
    
    Args:
        condition: Filter by condition (e.g., "diabetes")
        intervention: Filter by intervention (e.g., "metformin")
        max_results: Maximum number of results
        
    Returns:
        List of trial data dictionaries
    """
    client = get_client()
    
    query_parts = ["AREA[HasResults]true"]
    
    if condition:
        query_parts.append(f"AREA[Condition]{condition}")
    if intervention:
        query_parts.append(f"AREA[Intervention]{intervention}")
    
    query = " AND ".join(query_parts)
    
    return client.search_studies(query, max_results=max_results)


# =============================================================================
# DATA STRUCTURE CONVERSION
# =============================================================================

def convert_api_response_to_standard_format(api_response: Dict) -> Dict:
    """
    Convert ClinicalTrials.gov API v2 response to standard format.
    
    The API v2 returns data in a slightly different structure than bulk downloads.
    This function normalizes the format.
    
    Args:
        api_response: Raw API response
        
    Returns:
        Normalized trial data dictionary
    """
    # API v2 response structure is already compatible with our TrialData format
    # Just ensure required fields exist
    
    if 'protocolSection' not in api_response:
        # Try to extract from nested structure
        if 'study' in api_response:
            return api_response['study']
    
    return api_response


# =============================================================================
# VERSION HISTORY FUNCTIONS
# =============================================================================

def fetch_version_history(nct_id: str) -> List[Dict]:
    """
    Fetch version history for a trial.
    
    Args:
        nct_id: NCT identifier
        
    Returns:
        List of version metadata records
    """
    client = get_client()
    return client.get_study_versions(nct_id)


def fetch_all_versions_full(nct_id: str) -> List[Dict]:
    """
    Fetch full data for all versions of a trial.
    
    Args:
        nct_id: NCT identifier
        
    Returns:
        List of complete study data for each version
        
    Example:
        >>> versions = fetch_all_versions_full("NCT00256997")
        >>> for v in versions:
        ...     print(f"Version {v['_version_metadata']['version_number']}")
    """
    client = get_client()
    return client.get_all_study_versions_full(nct_id)


def fetch_trials_with_amendments(
    condition: Optional[str] = None,
    min_versions: int = 2,
    max_results: int = 100
) -> List[str]:
    """
    Find trials that have protocol amendments (multiple versions).
    
    Args:
        condition: Optional condition filter
        min_versions: Minimum number of versions
        max_results: Maximum number of trials to return
        
    Returns:
        List of NCT IDs with version history
    """
    client = get_client()
    
    query = ""
    if condition:
        query = f"AREA[Condition]{condition}"
    
    return client.get_studies_with_version_history(
        query=query,
        min_versions=min_versions,
        max_results=max_results
    )


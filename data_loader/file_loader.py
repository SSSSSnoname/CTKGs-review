"""
Local File Loader

Loads clinical trial data from local JSON files.
Supports large files with streaming JSON parsing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator

logger = logging.getLogger(__name__)


def load_trial_from_json(
    file_path: Union[str, Path],
    nct_id: str
) -> Dict:
    """
    Load a single trial from a JSON file containing multiple trials.
    
    Args:
        file_path: Path to JSON file
        nct_id: NCT identifier to find
        
    Returns:
        Trial data as dictionary
        
    Raises:
        ValueError: If trial not found in file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # For large files, use streaming parsing
    file_size = file_path.stat().st_size
    
    if file_size > 100 * 1024 * 1024:  # > 100 MB
        logger.info(f"Large file detected ({file_size / 1024 / 1024:.1f} MB), using streaming parser")
        return _load_trial_streaming(file_path, nct_id)
    else:
        return _load_trial_standard(file_path, nct_id)


def _load_trial_standard(file_path: Path, nct_id: str) -> Dict:
    """Load trial using standard JSON parsing (for smaller files)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different data structures
    trials = data if isinstance(data, list) else data.get('Studies', data.get('studies', [data]))
    
    for trial in trials:
        trial_nct_id = _extract_nct_id(trial)
        if trial_nct_id == nct_id:
            return trial
    
    raise ValueError(f"Trial {nct_id} not found in {file_path}")


def _load_trial_streaming(file_path: Path, nct_id: str) -> Dict:
    """Load trial using streaming JSON parsing (for large files)."""
    for trial in iterate_trials(file_path):
        trial_nct_id = _extract_nct_id(trial)
        if trial_nct_id == nct_id:
            return trial
    
    raise ValueError(f"Trial {nct_id} not found in {file_path}")


def _extract_nct_id(trial: Dict) -> str:
    """Extract NCT ID from trial data regardless of structure."""
    # Try protocolSection structure (API format)
    if 'protocolSection' in trial:
        return trial['protocolSection'].get('identificationModule', {}).get('nctId', '')
    
    # Try direct nctId field
    if 'nctId' in trial:
        return trial['nctId']
    
    # Try NCT field
    if 'NCT' in trial:
        return trial['NCT']
    
    # Try trials_base_information structure
    if 'trials_base_information' in trial:
        return trial['trials_base_information'].get('NCT', '')
    
    return ''


def load_trials_from_json(
    file_path: Union[str, Path],
    nct_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Load multiple trials from a JSON file.
    
    Args:
        file_path: Path to JSON file
        nct_ids: Optional list of NCT IDs to filter (None = load all)
        max_trials: Maximum number of trials to load
        
    Returns:
        Dictionary mapping NCT IDs to trial data
    """
    file_path = Path(file_path)
    nct_id_set = set(nct_ids) if nct_ids else None
    
    results = {}
    count = 0
    
    for trial in iterate_trials(file_path):
        trial_nct_id = _extract_nct_id(trial)
        
        if not trial_nct_id:
            continue
        
        if nct_id_set is None or trial_nct_id in nct_id_set:
            results[trial_nct_id] = trial
            count += 1
            
            if max_trials and count >= max_trials:
                break
            
            if nct_id_set and len(results) == len(nct_id_set):
                break
    
    logger.info(f"Loaded {len(results)} trials from {file_path}")
    return results


def iterate_trials(file_path: Union[str, Path]) -> Iterator[Dict]:
    """
    Iterate over trials in a JSON file using streaming parsing.
    
    This is memory-efficient for large files.
    
    Args:
        file_path: Path to JSON file
        
    Yields:
        Trial data dictionaries
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first character to determine structure
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Array of trials - use streaming JSON parser
            yield from _parse_json_array_streaming(f)
        elif first_char == '{':
            # Object with studies key or single trial
            data = json.load(f)
            if 'Studies' in data:
                yield from data['Studies']
            elif 'studies' in data:
                yield from data['studies']
            elif 'protocolSection' in data:
                yield data
            else:
                yield data


def _parse_json_array_streaming(file_handle) -> Iterator[Dict]:
    """
    Parse a JSON array in streaming fashion.
    
    This is a simplified streaming parser that works for arrays of objects.
    For very large files, consider using ijson library.
    """
    content = file_handle.read()
    
    # Find array boundaries
    if not content.startswith('['):
        raise ValueError("Expected JSON array")
    
    # Track bracket depth to find object boundaries
    depth = 0
    start = None
    in_string = False
    escape = False
    
    for i, c in enumerate(content):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(content[start:i+1])
                    yield obj
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse object at position {start}: {e}")
                start = None


def get_trial_count(file_path: Union[str, Path]) -> int:
    """
    Count the number of trials in a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Number of trials in the file
    """
    count = 0
    for _ in iterate_trials(file_path):
        count += 1
    return count


def get_nct_ids_from_file(
    file_path: Union[str, Path],
    max_ids: Optional[int] = None
) -> List[str]:
    """
    Extract all NCT IDs from a JSON file.
    
    Args:
        file_path: Path to JSON file
        max_ids: Maximum number of IDs to return
        
    Returns:
        List of NCT IDs
    """
    nct_ids = []
    
    for trial in iterate_trials(file_path):
        nct_id = _extract_nct_id(trial)
        if nct_id:
            nct_ids.append(nct_id)
            
            if max_ids and len(nct_ids) >= max_ids:
                break
    
    return nct_ids


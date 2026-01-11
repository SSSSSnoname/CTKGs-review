"""
Task 7: Building the Dynamic CTKG (Protocol Versioning)

For trials with multiple registry versions:
- Each version is represented as a distinct node (trial ID + version)
- 24 harmonized fields tracked (locations, enrollment, outcomes, key dates, etc.)
- Version-to-version differences computed and classified as: added, removed, modified, unchanged

Text-intensive fields (eligibility criteria, outcome definitions, termination reasons)
are structured via LLM parsing for entity-level comparison.

Termination reasons are normalized to standard categories.
All versioned nodes link back to canonical NCTID for trial-centric CTKG consistency.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import re
from datetime import datetime

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ChangeType(Enum):
    """Types of changes between versions."""
    FIRST_VERSION = "first_version"  # First version, no comparison available
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class FieldCategory(Enum):
    """Categories of tracked fields."""
    STATUS = "status"
    ENROLLMENT = "enrollment"
    DESIGN = "design"
    LOCATIONS = "locations"
    INTERVENTIONS = "interventions"
    OUTCOMES = "outcomes"
    ELIGIBILITY = "eligibility"
    SPONSORS = "sponsors"
    DESCRIPTION = "description"
    DATES = "dates"


# Standard termination reason categories with descriptions for LLM classification
TERMINATION_CATEGORIES = {
    "STRATEGIC_DECISION": "Strategic or business decisions including reprioritization, corporate restructuring, portfolio changes, or sponsor decisions unrelated to trial performance",
    "RECRUITMENT_DIFFICULTIES": "Challenges with patient/subject recruitment, enrollment, or accrual including slow enrollment, inability to enroll sufficient participants",
    "LACK_OF_EFFICACY": "Insufficient efficacy, futility analysis results, failure to demonstrate benefit, or not meeting primary endpoints",
    "SAFETY_CONCERNS": "Safety issues including adverse events, toxicity, safety signals, unacceptable risk-benefit ratio",
    "ADMINISTRATIVE": "Administrative or operational issues including site closures, investigator issues, procedural problems, logistical challenges",
    "FUNDING_ISSUES": "Financial problems including lack of funding, budget constraints, insufficient resources",
    "REGULATORY": "Regulatory issues including FDA actions, approval problems, IND issues, compliance concerns",
    "COMPLETED_EARLY": "Trial completed earlier than planned due to meeting endpoints early, reaching target outcomes, or early success",
    "COVID_IMPACT": "Impact of COVID-19 pandemic including lockdowns, disruptions to trial operations, or pandemic-related resource constraints",
    "OTHER": "Termination reasons that do not fit into the above categories"
}

# List of valid termination category names
TERMINATION_CATEGORY_NAMES = list(TERMINATION_CATEGORIES.keys())


# 24 Harmonized tracked fields organized by category
TRACKED_FIELDS_CONFIG = {
    # Status fields
    'overallStatus': {
        'category': FieldCategory.STATUS,
        'path': ['statusModule', 'overallStatus'],
        'description': 'Overall trial status'
    },
    'whyStopped': {
        'category': FieldCategory.STATUS,
        'path': ['statusModule', 'whyStopped'],
        'description': 'Reason trial was stopped'
    },
    'startDate': {
        'category': FieldCategory.DATES,
        'path': ['statusModule', 'startDateStruct', 'date'],
        'description': 'Study start date'
    },
    'completionDate': {
        'category': FieldCategory.DATES,
        'path': ['statusModule', 'completionDateStruct', 'date'],
        'description': 'Study completion date'
    },
    'primaryCompletionDate': {
        'category': FieldCategory.DATES,
        'path': ['statusModule', 'primaryCompletionDateStruct', 'date'],
        'description': 'Primary completion date'
    },
    
    # Enrollment fields
    'enrollmentCount': {
        'category': FieldCategory.ENROLLMENT,
        'path': ['designModule', 'enrollmentInfo', 'count'],
        'description': 'Target enrollment number'
    },
    'enrollmentType': {
        'category': FieldCategory.ENROLLMENT,
        'path': ['designModule', 'enrollmentInfo', 'type'],
        'description': 'Enrollment type (actual/anticipated)'
    },
    
    # Design fields
    'phase': {
        'category': FieldCategory.DESIGN,
        'path': ['designModule', 'phases'],
        'description': 'Trial phase(s)'
    },
    'studyType': {
        'category': FieldCategory.DESIGN,
        'path': ['designModule', 'studyType'],
        'description': 'Study type (interventional/observational)'
    },
    'allocation': {
        'category': FieldCategory.DESIGN,
        'path': ['designModule', 'designInfo', 'allocation'],
        'description': 'Allocation method'
    },
    'interventionModel': {
        'category': FieldCategory.DESIGN,
        'path': ['designModule', 'designInfo', 'interventionModel'],
        'description': 'Intervention model'
    },
    'masking': {
        'category': FieldCategory.DESIGN,
        'path': ['designModule', 'designInfo', 'maskingInfo'],
        'description': 'Masking/blinding information'
    },
    
    # Location fields
    'locations': {
        'category': FieldCategory.LOCATIONS,
        'path': ['contactsLocationsModule', 'locations'],
        'description': 'Study locations',
        'is_list': True
    },
    'centralContacts': {
        'category': FieldCategory.LOCATIONS,
        'path': ['contactsLocationsModule', 'centralContacts'],
        'description': 'Central contact information',
        'is_list': True
    },
    
    # Intervention fields
    'interventions': {
        'category': FieldCategory.INTERVENTIONS,
        'path': ['armsInterventionsModule', 'interventions'],
        'description': 'Intervention details',
        'is_list': True
    },
    'armGroups': {
        'category': FieldCategory.INTERVENTIONS,
        'path': ['armsInterventionsModule', 'armGroups'],
        'description': 'Arm groups',
        'is_list': True
    },
    
    # Outcome fields
    'primaryOutcomes': {
        'category': FieldCategory.OUTCOMES,
        'path': ['outcomesModule', 'primaryOutcomes'],
        'description': 'Primary outcome measures',
        'is_list': True,
        'text_intensive': True
    },
    'secondaryOutcomes': {
        'category': FieldCategory.OUTCOMES,
        'path': ['outcomesModule', 'secondaryOutcomes'],
        'description': 'Secondary outcome measures',
        'is_list': True,
        'text_intensive': True
    },
    
    # Eligibility fields
    'eligibilityCriteria': {
        'category': FieldCategory.ELIGIBILITY,
        'path': ['eligibilityModule', 'eligibilityCriteria'],
        'description': 'Inclusion/exclusion criteria text',
        'text_intensive': True
    },
    'eligibilityMinAge': {
        'category': FieldCategory.ELIGIBILITY,
        'path': ['eligibilityModule', 'minimumAge'],
        'description': 'Minimum age'
    },
    'eligibilityMaxAge': {
        'category': FieldCategory.ELIGIBILITY,
        'path': ['eligibilityModule', 'maximumAge'],
        'description': 'Maximum age'
    },
    'eligibilitySex': {
        'category': FieldCategory.ELIGIBILITY,
        'path': ['eligibilityModule', 'sex'],
        'description': 'Sex eligibility'
    },
    
    # Sponsor fields
    'sponsors': {
        'category': FieldCategory.SPONSORS,
        'path': ['sponsorCollaboratorsModule', 'leadSponsor'],
        'description': 'Lead sponsor'
    },
    'collaborators': {
        'category': FieldCategory.SPONSORS,
        'path': ['sponsorCollaboratorsModule', 'collaborators'],
        'description': 'Collaborators',
        'is_list': True
    },
    
    # Description fields
    'briefSummary': {
        'category': FieldCategory.DESCRIPTION,
        'path': ['descriptionModule', 'briefSummary'],
        'description': 'Brief summary',
        'text_intensive': True
    },
    'detailedDescription': {
        'category': FieldCategory.DESCRIPTION,
        'path': ['descriptionModule', 'detailedDescription'],
        'description': 'Detailed description',
        'text_intensive': True
    },
    
    # Additional date fields
    'studyFirstPostDate': {
        'category': FieldCategory.DATES,
        'path': ['statusModule', 'studyFirstPostDateStruct', 'date'],
        'description': 'First posted date'
    },
    'lastUpdatePostDate': {
        'category': FieldCategory.DATES,
        'path': ['statusModule', 'lastUpdatePostDateStruct', 'date'],
        'description': 'Last update posted date'
    },
    
    # Conditions
    'conditions': {
        'category': FieldCategory.STATUS,
        'path': ['conditionsModule', 'conditions'],
        'description': 'Studied conditions',
        'is_list': True
    },
}

# List of all tracked field names (24 fields)
TRACKED_FIELDS = list(TRACKED_FIELDS_CONFIG.keys())


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TaggedEntity:
    """
    A single entity within a list field with its change status.
    
    For example, in a locations list, each location is a TaggedEntity.
    """
    value: Any  # The entity value (e.g., a location dict)
    status: ChangeType  # added, removed, modified, unchanged, first_version
    entity_key: str = ""  # Human-readable key for the entity
    previous_value: Any = None  # Only for modified entities
    
    def to_dict(self) -> Dict:
        result = {
            'value': self.value,
            'status': self.status.value,
            'entity_key': self.entity_key
        }
        if self.status == ChangeType.MODIFIED and self.previous_value is not None:
            result['previous_value'] = self.previous_value
        return result


@dataclass
class TaggedField:
    """
    A field with its value and change status relative to the previous version.
    
    Each field in a version node is tagged with:
    - value: Current value of the field
    - status: Change status (first_version, added, removed, modified, unchanged)
    - previous_value: Value from previous version (only for modified fields)
    - entity_changes: For list fields, individual entity-level change tracking
    """
    field_name: str
    value: Any
    status: ChangeType
    previous_value: Any = None
    category: Optional[FieldCategory] = None
    entity_changes: List[TaggedEntity] = field(default_factory=list)  # Entity-level tracking for list fields
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            'value': self.value,
            'status': self.status.value
        }
        if self.status == ChangeType.MODIFIED:
            result['previous_value'] = self.previous_value
        if self.category:
            result['category'] = self.category.value
        # Include entity-level changes for list fields
        if self.entity_changes:
            result['entity_changes'] = [e.to_dict() for e in self.entity_changes]
        return result
    
    @classmethod
    def from_dict(cls, field_name: str, data: Dict) -> 'TaggedField':
        """Create from dictionary."""
        entity_changes = []
        if 'entity_changes' in data:
            for ec in data['entity_changes']:
                entity_changes.append(TaggedEntity(
                    value=ec.get('value'),
                    status=ChangeType(ec.get('status', 'unchanged')),
                    entity_key=ec.get('entity_key', ''),
                    previous_value=ec.get('previous_value')
                ))
        
        return cls(
            field_name=field_name,
            value=data.get('value'),
            status=ChangeType(data.get('status', 'unchanged')),
            previous_value=data.get('previous_value'),
            category=FieldCategory(data['category']) if data.get('category') else None,
            entity_changes=entity_changes
        )


@dataclass
class VersionNode:
    """
    Represents a specific version of a clinical trial.
    
    Each version is encoded as a distinct node with the format: {NCTID}_v{version_number}
    
    Fields can be accessed in two ways:
    - tracked_fields: Raw field values (Dict[str, Any])
    - tagged_fields: Fields with change status tags (Dict[str, TaggedField])
    """
    node_id: str  # "{NCTID}_v{version_number}"
    nct_id: str   # Canonical NCTID for linking back to trial-centric CTKG
    version_number: int
    version_date: Optional[str] = None
    tracked_fields: Dict[str, Any] = field(default_factory=dict)
    tagged_fields: Dict[str, TaggedField] = field(default_factory=dict)  # Fields with status tags
    structured_text_fields: Dict[str, Any] = field(default_factory=dict)
    tagged_structured_text_fields: Dict[str, Any] = field(default_factory=dict)  # Structured text with status tags
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'node_id': self.node_id,
            'nct_id': self.nct_id,
            'version_number': self.version_number,
            'version_date': self.version_date,
            'tracked_fields': self.tracked_fields,
            'tagged_fields': {
                name: tf.to_dict() for name, tf in self.tagged_fields.items()
            },
            'structured_text_fields': self.structured_text_fields,
            'tagged_structured_text_fields': self.tagged_structured_text_fields,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionNode':
        """Create from dictionary."""
        tagged = {}
        if 'tagged_fields' in data:
            for name, tf_data in data['tagged_fields'].items():
                tagged[name] = TaggedField.from_dict(name, tf_data)
        
        return cls(
            node_id=data.get('node_id', ''),
            nct_id=data.get('nct_id', ''),
            version_number=data.get('version_number', 0),
            version_date=data.get('version_date'),
            tracked_fields=data.get('tracked_fields', {}),
            tagged_fields=tagged,
            structured_text_fields=data.get('structured_text_fields', {}),
            tagged_structured_text_fields=data.get('tagged_structured_text_fields', {}),
            metadata=data.get('metadata', {})
        )
    
    def get_field_status_summary(self) -> Dict[str, int]:
        """Get count of fields by status."""
        summary = {status.value: 0 for status in ChangeType}
        for tf in self.tagged_fields.values():
            summary[tf.status.value] += 1
        return summary


@dataclass
class FieldChange:
    """Represents a change in a specific field between versions."""
    field_name: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    category: Optional[FieldCategory] = None
    entity_level_changes: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'field_name': self.field_name,
            'change_type': self.change_type.value,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'category': self.category.value if self.category else None,
            'entity_level_changes': self.entity_level_changes
        }


@dataclass
class VersionChangeRelation:
    """Represents the relationship between two adjacent versions."""
    from_version: str  # "{NCTID}_v{n}"
    to_version: str    # "{NCTID}_v{n+1}"
    nct_id: str
    from_version_num: int
    to_version_num: int
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    field_changes: List[FieldChange] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'from_version': self.from_version,
            'to_version': self.to_version,
            'nct_id': self.nct_id,
            'from_version_num': self.from_version_num,
            'to_version_num': self.to_version_num,
            'from_date': self.from_date,
            'to_date': self.to_date,
            'field_changes': [fc.to_dict() for fc in self.field_changes],
            'summary': self.summary
        }


@dataclass 
class TerminationInfo:
    """
    Normalized termination information.
    
    Termination reasons are classified by LLM into standard categories.
    """
    status: str                          # Trial status (TERMINATED, WITHDRAWN, etc.)
    category: str                        # Normalized category from TERMINATION_CATEGORIES
    original_text: Optional[str] = None  # Original free-text termination reason
    confidence: float = 1.0              # Classification confidence (0-1)
    category_description: Optional[str] = None  # Description of the assigned category
    
    def to_dict(self) -> Dict:
        result = {
            'status': self.status,
            'category': self.category,
            'original_text': self.original_text,
            'confidence': self.confidence
        }
        if self.category_description:
            result['category_description'] = self.category_description
        return result


# =============================================================================
# FIELD EXTRACTOR
# =============================================================================

class VersionFieldExtractor:
    """
    Extracts and harmonizes 24 tracked fields from version data.
    
    Handles nested paths in ClinicalTrials.gov API response structure.
    """
    
    def __init__(self, field_config: Dict = None):
        """
        Initialize the field extractor.
        
        Args:
            field_config: Configuration for tracked fields
        """
        self.field_config = field_config or TRACKED_FIELDS_CONFIG
    
    def extract_all_fields(self, version_data: Dict) -> Dict[str, Any]:
        """
        Extract all 24 tracked fields from version data.
        
        Args:
            version_data: Raw version data from API
            
        Returns:
            Dictionary of field name -> extracted value
        """
        extracted = {}
        
        for field_name, config in self.field_config.items():
            value = self.extract_field(version_data, field_name)
            if value is not None:
                extracted[field_name] = value
        
        return extracted
    
    def extract_field(self, version_data: Dict, field_name: str) -> Any:
        """
        Extract a specific field from version data.
        
        Args:
            version_data: Raw version data
            field_name: Name of field to extract
            
        Returns:
            Extracted value or None
        """
        if field_name not in self.field_config:
            return None
        
        config = self.field_config[field_name]
        path = config.get('path', [])
        
        # Handle protocolSection wrapper
        if 'protocolSection' in version_data:
            root = version_data['protocolSection']
        else:
            root = version_data
        
        return self._navigate_path(root, path)
    
    def _navigate_path(self, data: Any, path: List[str]) -> Any:
        """Navigate a nested path to extract a value."""
        current = data
        
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                current = current[idx] if idx < len(current) else None
            else:
                return None
            
            if current is None:
                return None
        
        return current
    
    def get_field_category(self, field_name: str) -> Optional[FieldCategory]:
        """Get the category of a field."""
        if field_name in self.field_config:
            return self.field_config[field_name].get('category')
        return None
    
    def is_text_intensive(self, field_name: str) -> bool:
        """Check if a field is text-intensive (needs LLM parsing)."""
        if field_name in self.field_config:
            return self.field_config[field_name].get('text_intensive', False)
        return False
    
    def is_list_field(self, field_name: str) -> bool:
        """Check if a field contains a list."""
        if field_name in self.field_config:
            return self.field_config[field_name].get('is_list', False)
        return False


# =============================================================================
# VERSION DIFF ENGINE
# =============================================================================

class VersionDiffEngine:
    """
    Computes differences between adjacent trial versions.
    
    Classifies changes as: added, removed, modified, unchanged.
    Supports both field-level and entity-level comparison.
    """
    
    def __init__(self, field_extractor: VersionFieldExtractor = None):
        """
        Initialize the diff engine.
        
        Args:
            field_extractor: Field extractor instance
        """
        self.field_extractor = field_extractor or VersionFieldExtractor()
    
    def compute_diff(
        self, 
        version1: Dict, 
        version2: Dict,
        nct_id: str,
        v1_num: int,
        v2_num: int
    ) -> VersionChangeRelation:
        """
        Compute differences between two versions.
        
        Args:
            version1: First version data
            version2: Second version data  
            nct_id: NCT identifier
            v1_num: First version number
            v2_num: Second version number
            
        Returns:
            VersionChangeRelation with all field changes
        """
        field_changes = []
        change_counts = {
            'added': 0,
            'removed': 0,
            'modified': 0,
            'unchanged': 0
        }
        
        for field_name in TRACKED_FIELDS:
            val1 = self.field_extractor.extract_field(version1, field_name)
            val2 = self.field_extractor.extract_field(version2, field_name)
            
            change_type = self._classify_change(val1, val2)
            change_counts[change_type.value] += 1
            
            if change_type != ChangeType.UNCHANGED:
                # Get entity-level changes for list/text fields
                entity_changes = []
                if self.field_extractor.is_list_field(field_name):
                    entity_changes = self._compute_list_diff(val1, val2)
                
                field_change = FieldChange(
                    field_name=field_name,
                    change_type=change_type,
                    old_value=val1,
                    new_value=val2,
                    category=self.field_extractor.get_field_category(field_name),
                    entity_level_changes=entity_changes
                )
                field_changes.append(field_change)
        
        # Get version dates
        v1_date = version1.get('_version_metadata', {}).get('version_date')
        v2_date = version2.get('_version_metadata', {}).get('version_date')
        
        return VersionChangeRelation(
            from_version=f"{nct_id}_v{v1_num}",
            to_version=f"{nct_id}_v{v2_num}",
            nct_id=nct_id,
            from_version_num=v1_num,
            to_version_num=v2_num,
            from_date=v1_date,
            to_date=v2_date,
            field_changes=field_changes,
            summary={
                'total_changes': len(field_changes),
                'added': change_counts['added'],
                'removed': change_counts['removed'],
                'modified': change_counts['modified'],
                'unchanged': change_counts['unchanged']
            }
        )
    
    def _classify_change(self, val1: Any, val2: Any) -> ChangeType:
        """
        Classify the type of change between two values.
        
        Handles edge cases:
        - Empty list [], empty string "", None are treated as "no value"
        - [data] -> [] is REMOVED (not MODIFIED)
        - [] -> [data] is ADDED (not MODIFIED)
        """
        # Normalize empty values to None for comparison
        val1_empty = self._is_empty(val1)
        val2_empty = self._is_empty(val2)
        
        if val1_empty and val2_empty:
            return ChangeType.UNCHANGED
        elif val1_empty and not val2_empty:
            return ChangeType.ADDED
        elif not val1_empty and val2_empty:
            return ChangeType.REMOVED
        elif self._values_equal(val1, val2):
            return ChangeType.UNCHANGED
        else:
            return ChangeType.MODIFIED
    
    def _is_empty(self, val: Any) -> bool:
        """
        Check if a value is considered empty (None, [], "", {}).
        
        Empty values are treated as "no value" for change classification.
        """
        if val is None:
            return True
        if isinstance(val, (list, dict, str)) and len(val) == 0:
            return True
        return False
    
    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal (handles complex types)."""
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, dict):
            return self._dicts_equal(val1, val2)
        elif isinstance(val1, list):
            return self._lists_equal(val1, val2)
        else:
            return val1 == val2
    
    def _dicts_equal(self, d1: Dict, d2: Dict) -> bool:
        """Check if two dictionaries are equal."""
        if set(d1.keys()) != set(d2.keys()):
            return False
        return all(self._values_equal(d1[k], d2[k]) for k in d1.keys())
    
    def _lists_equal(self, l1: List, l2: List) -> bool:
        """Check if two lists are equal."""
        if len(l1) != len(l2):
            return False
        return all(self._values_equal(v1, v2) for v1, v2 in zip(l1, l2))
    
    def _compute_list_diff(
        self, 
        list1: Optional[List], 
        list2: Optional[List]
    ) -> List[Dict]:
        """
        Compute entity-level differences for list fields.
        
        Returns list of entity changes with type (added/removed/modified).
        """
        if list1 is None:
            list1 = []
        if list2 is None:
            list2 = []
        
        entity_changes = []
        
        # Convert to comparable format
        items1 = self._list_to_comparable(list1)
        items2 = self._list_to_comparable(list2)
        
        # Find added items
        for item in items2:
            if item not in items1:
                entity_changes.append({
                    'type': 'added',
                    'entity': self._get_entity_key(list2[items2.index(item)])
                })
        
        # Find removed items
        for item in items1:
            if item not in items2:
                entity_changes.append({
                    'type': 'removed',
                    'entity': self._get_entity_key(list1[items1.index(item)])
                })
        
        return entity_changes
    
    def _list_to_comparable(self, lst: List) -> List[str]:
        """Convert list items to comparable string format."""
        result = []
        for item in lst:
            if isinstance(item, dict):
                # Use JSON for consistent comparison
                result.append(json.dumps(item, sort_keys=True))
            else:
                result.append(str(item))
        return result
    
    def _get_entity_key(self, item: Any) -> str:
        """Extract a readable key from an entity."""
        if isinstance(item, dict):
            # Try common key fields
            for key in ['name', 'title', 'measure', 'city', 'facility']:
                if key in item:
                    return str(item[key])
            return json.dumps(item)[:100]
        return str(item)


# =============================================================================
# TEXT FIELD STRUCTURER (LLM-based)
# =============================================================================

class TextFieldStructurer:
    """
    Structures text-intensive fields using LLM parsing.
    
    Handles:
    - Eligibility criteria parsing
    - Outcome definitions parsing
    - Termination reason normalization
    """
    
    def __init__(self, llm_caller=None):
        """
        Initialize the structurer.
        
        Args:
            llm_caller: Callable for LLM invocation
        """
        self.llm_caller = llm_caller
    
    def structure_eligibility(self, raw_text: str) -> Dict:
        """
        Parse eligibility criteria into structured format.
        
        Args:
            raw_text: Raw eligibility criteria text
            
        Returns:
            Structured eligibility with inclusion/exclusion lists
        """
        if not raw_text:
            return {'inclusion': [], 'exclusion': []}
        
        # Try rule-based parsing first
        structured = self._parse_eligibility_rules(raw_text)
        
        # If LLM available and rule-based insufficient, use LLM
        if self.llm_caller and len(structured['inclusion']) + len(structured['exclusion']) < 3:
            try:
                structured = self._parse_eligibility_llm(raw_text)
            except Exception as e:
                logger.warning(f"LLM eligibility parsing failed: {e}")
        
        return structured
    
    def _parse_eligibility_rules(self, text: str) -> Dict:
        """Rule-based eligibility parsing."""
        inclusion = []
        exclusion = []
        
        # Split by common section headers
        text_lower = text.lower()
        
        # Find inclusion section
        inc_patterns = [
            r'inclusion criteria[:\s]*\n?(.*?)(?=exclusion criteria|\Z)',
            r'key inclusion[:\s]*\n?(.*?)(?=exclusion|\Z)',
        ]
        
        for pattern in inc_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                inc_text = match.group(1)
                inclusion = self._extract_criteria_items(inc_text)
                break
        
        # Find exclusion section
        exc_patterns = [
            r'exclusion criteria[:\s]*\n?(.*?)(?=\Z)',
            r'key exclusion[:\s]*\n?(.*?)(?=\Z)',
        ]
        
        for pattern in exc_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                exc_text = match.group(1)
                exclusion = self._extract_criteria_items(exc_text)
                break
        
        return {'inclusion': inclusion, 'exclusion': exclusion}
    
    def _extract_criteria_items(self, text: str) -> List[str]:
        """Extract individual criteria items from text."""
        items = []
        
        # Split by bullet points, numbers, or newlines
        lines = re.split(r'\n\s*[-•*]|\n\s*\d+[.)]\s*|\n{2,}', text)
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Skip very short items
                items.append(line)
        
        return items
    
    def _parse_eligibility_llm(self, text: str) -> Dict:
        """LLM-based eligibility parsing."""
        prompt = f"""Parse the following clinical trial eligibility criteria into structured format.
        
Extract each criterion as a separate item.

Eligibility Criteria:
{text[:3000]}

Return a JSON object with this format:
{{
    "inclusion": ["criterion 1", "criterion 2", ...],
    "exclusion": ["criterion 1", "criterion 2", ...]
}}

Return ONLY valid JSON."""
        
        response = self.llm_caller(prompt)
        
        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        return {'inclusion': [], 'exclusion': []}
    
    def structure_outcomes(self, outcomes: List[Dict]) -> List[Dict]:
        """
        Structure outcome definitions for comparison.
        
        Args:
            outcomes: Raw outcome data
            
        Returns:
            Structured outcomes with extracted components
        """
        structured = []
        
        for outcome in outcomes:
            structured_outcome = {
                'measure': outcome.get('measure', ''),
                'description': outcome.get('description', ''),
                'timeFrame': outcome.get('timeFrame', ''),
                # Normalize measure name for comparison
                'normalized_measure': self._normalize_outcome_name(
                    outcome.get('measure', '')
                )
            }
            structured.append(structured_outcome)
        
        return structured
    
    def _normalize_outcome_name(self, name: str) -> str:
        """Normalize outcome name for comparison."""
        if not name:
            return ''
        
        # Lowercase, remove extra whitespace
        normalized = ' '.join(name.lower().split())
        
        # Remove common variations
        normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def compare_structured_eligibility(
        self, 
        v1_eligibility: Dict, 
        v2_eligibility: Dict
    ) -> Dict:
        """
        Compare structured eligibility between versions.
        
        Returns entity-level changes.
        """
        changes = {
            'inclusion': {'added': [], 'removed': [], 'modified': []},
            'exclusion': {'added': [], 'removed': [], 'modified': []}
        }
        
        for criteria_type in ['inclusion', 'exclusion']:
            v1_items = set(v1_eligibility.get(criteria_type, []))
            v2_items = set(v2_eligibility.get(criteria_type, []))
            
            changes[criteria_type]['added'] = list(v2_items - v1_items)
            changes[criteria_type]['removed'] = list(v1_items - v2_items)
        
        return changes
    
    def compare_structured_outcomes(
        self,
        v1_outcomes: List[Dict],
        v2_outcomes: List[Dict]
    ) -> Dict:
        """
        Compare structured outcomes between versions.
        
        Returns entity-level changes.
        """
        v1_measures = {o['normalized_measure']: o for o in v1_outcomes}
        v2_measures = {o['normalized_measure']: o for o in v2_outcomes}
        
        v1_keys = set(v1_measures.keys())
        v2_keys = set(v2_measures.keys())
        
        return {
            'added': [v2_measures[k]['measure'] for k in (v2_keys - v1_keys)],
            'removed': [v1_measures[k]['measure'] for k in (v1_keys - v2_keys)],
            'modified': []  # Would need deeper comparison
        }
    
    def tag_structured_eligibility(
        self,
        current_eligibility: Dict,
        previous_eligibility: Optional[Dict] = None
    ) -> Dict:
        """
        Create tagged version of structured eligibility with status for each criterion.
        
        Args:
            current_eligibility: Current version's structured eligibility
            previous_eligibility: Previous version's structured eligibility (None for first version)
            
        Returns:
            Tagged eligibility with status for each criterion
        """
        if previous_eligibility is None:
            # First version - all items are "first_version"
            return {
                'inclusion': [
                    {'value': item, 'status': 'first_version'}
                    for item in current_eligibility.get('inclusion', [])
                ],
                'exclusion': [
                    {'value': item, 'status': 'first_version'}
                    for item in current_eligibility.get('exclusion', [])
                ]
            }
        
        tagged = {'inclusion': [], 'exclusion': []}
        
        for criteria_type in ['inclusion', 'exclusion']:
            prev_items = set(previous_eligibility.get(criteria_type, []))
            curr_items = current_eligibility.get(criteria_type, [])
            
            for item in curr_items:
                if item in prev_items:
                    tagged[criteria_type].append({'value': item, 'status': 'unchanged'})
                else:
                    # Check if it's a modification (similar but not exact match)
                    is_modified = False
                    for prev_item in prev_items:
                        if self._is_similar_criterion(item, prev_item):
                            tagged[criteria_type].append({
                                'value': item, 
                                'status': 'modified',
                                'previous_value': prev_item
                            })
                            is_modified = True
                            break
                    if not is_modified:
                        tagged[criteria_type].append({'value': item, 'status': 'added'})
            
            # Add removed items (from previous but not in current)
            for prev_item in prev_items:
                if prev_item not in curr_items:
                    # Check if modified (already handled above)
                    is_modified = any(
                        self._is_similar_criterion(curr_item, prev_item) 
                        for curr_item in curr_items
                    )
                    if not is_modified:
                        tagged[criteria_type].append({
                            'value': prev_item, 
                            'status': 'removed'
                        })
        
        return tagged
    
    def _is_similar_criterion(self, item1: str, item2: str) -> bool:
        """
        Check if two criteria are similar (likely a modification of each other).
        Uses simple heuristics: same start, similar length, etc.
        """
        if not item1 or not item2:
            return False
        
        # Normalize for comparison
        norm1 = ' '.join(item1.lower().split())
        norm2 = ' '.join(item2.lower().split())
        
        # If one is a prefix of the other (at least 50% overlap)
        min_len = min(len(norm1), len(norm2))
        prefix_len = min_len // 2
        
        if norm1[:prefix_len] == norm2[:prefix_len]:
            # Check length similarity (within 50%)
            len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            if len_ratio > 0.5:
                return True
        
        return False
    
    def tag_structured_outcomes(
        self,
        current_outcomes: List[Dict],
        previous_outcomes: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Create tagged version of structured outcomes with status for each outcome.
        
        Args:
            current_outcomes: Current version's structured outcomes
            previous_outcomes: Previous version's structured outcomes (None for first version)
            
        Returns:
            Tagged outcomes with status for each outcome
        """
        if previous_outcomes is None:
            # First version - all items are "first_version"
            return [
                {**outcome, 'status': 'first_version'}
                for outcome in current_outcomes
            ]
        
        # Build lookup by normalized measure
        prev_measures = {o.get('normalized_measure', ''): o for o in previous_outcomes}
        prev_keys = set(prev_measures.keys())
        
        tagged = []
        seen_prev_keys = set()
        
        for outcome in current_outcomes:
            norm_measure = outcome.get('normalized_measure', '')
            
            if norm_measure in prev_keys:
                # Check if other fields changed
                prev = prev_measures[norm_measure]
                if self._outcomes_equal(outcome, prev):
                    tagged.append({**outcome, 'status': 'unchanged'})
                else:
                    tagged.append({
                        **outcome, 
                        'status': 'modified',
                        'previous_value': prev
                    })
                seen_prev_keys.add(norm_measure)
            else:
                tagged.append({**outcome, 'status': 'added'})
        
        # Add removed outcomes
        for key in prev_keys - seen_prev_keys:
            tagged.append({
                **prev_measures[key],
                'status': 'removed'
            })
        
        return tagged
    
    def _outcomes_equal(self, o1: Dict, o2: Dict) -> bool:
        """Check if two outcomes are equal (ignoring normalized fields)."""
        return (
            o1.get('measure') == o2.get('measure') and
            o1.get('timeFrame') == o2.get('timeFrame') and
            o1.get('description') == o2.get('description')
        )


# =============================================================================
# TERMINATION NORMALIZER
# =============================================================================

class TerminationNormalizer:
    """
    Normalizes free-text termination reasons to standard categories using LLM.
    
    Categories are defined with descriptions to guide LLM classification.
    """
    
    def __init__(self, llm_caller=None):
        """
        Initialize the normalizer.
        
        Args:
            llm_caller: Callable for LLM invocation (required for classification)
        """
        self.llm_caller = llm_caller
        self.categories = TERMINATION_CATEGORIES
        self.category_names = TERMINATION_CATEGORY_NAMES
    
    def normalize(self, why_stopped: str, status: str = None) -> TerminationInfo:
        """
        Normalize termination reason to standard category using LLM.
        
        Args:
            why_stopped: Free-text termination reason
            status: Overall trial status
            
        Returns:
            TerminationInfo with normalized category
        """
        if not why_stopped:
            return TerminationInfo(
                status=status or 'UNKNOWN',
                category='OTHER',
                original_text=None,
                confidence=0.5
            )
        
        # Use LLM to classify termination reason
        if self.llm_caller:
            try:
                category, confidence = self._classify_with_llm(why_stopped)
                return TerminationInfo(
                    status=status or 'TERMINATED',
                    category=category,
                    original_text=why_stopped,
                    confidence=confidence,
                    category_description=self.categories.get(category)
                )
            except Exception as e:
                logger.warning(f"LLM termination classification failed: {e}")
                # Fall back to OTHER if LLM fails
                return TerminationInfo(
                    status=status or 'TERMINATED',
                    category='OTHER',
                    original_text=why_stopped,
                    confidence=0.3,
                    category_description=self.categories.get('OTHER')
                )
        else:
            # No LLM available - return OTHER with low confidence
            logger.warning("No LLM available for termination classification, using OTHER")
            return TerminationInfo(
                status=status or 'TERMINATED',
                category='OTHER',
                original_text=why_stopped,
                confidence=0.3,
                category_description=self.categories.get('OTHER')
            )
    
    def _classify_with_llm(self, text: str) -> Tuple[str, float]:
        """
        Use LLM to classify termination reason into a category.
        
        Args:
            text: Free-text termination reason
            
        Returns:
            Tuple of (category_name, confidence_score)
        """
        # Build prompt with category descriptions
        categories_with_desc = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in self.categories.items()
        ])
        
        prompt = f"""You are a clinical trial expert. Classify the following termination reason into exactly ONE of the categories below.

CATEGORIES:
{categories_with_desc}

TERMINATION REASON: "{text}"

Instructions:
1. Analyze the termination reason carefully
2. Match it to the most appropriate category based on the descriptions
3. Return ONLY the category name (e.g., SAFETY_CONCERNS), nothing else

Category:"""
        
        response = self.llm_caller(prompt)
        category = response.strip().upper().replace(' ', '_').replace('-', '_')
        
        # Clean up response - extract just the category name
        for cat_name in self.category_names:
            if cat_name in category:
                return cat_name, 0.9
        
        # If no exact match found, try to find partial match
        category_clean = category.split('\n')[0].strip()
        if category_clean in self.category_names:
            return category_clean, 0.9
        
        # Default to OTHER if can't parse response
        logger.warning(f"Could not parse LLM category response: {response}")
        return 'OTHER', 0.5
    
    def get_category_description(self, category: str) -> str:
        """Get the description for a termination category."""
        return self.categories.get(category, "Unknown category")


# =============================================================================
# MAIN TASK HANDLER
# =============================================================================

class Task7DynamicCTKG(BaseTaskHandler):
    """
    Task 7: Building the Dynamic CTKG
    
    Tracks protocol modifications and version histories,
    enabling analysis of trial evolution over time.
    
    Features:
    - Version node creation with 24 harmonized fields
    - Version-to-version diff computation
    - Entity-level comparison for text-intensive fields
    - Termination reason normalization
    - Integration with trial-centric CTKG via canonical NCTID
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_extractor = VersionFieldExtractor()
        self.diff_engine = VersionDiffEngine(self.field_extractor)
        self.text_structurer = None  # Initialized lazily with LLM
        self.termination_normalizer = None  # Initialized lazily with LLM
    
    @property
    def task_name(self) -> str:
        return "task7_dynamic_ctkg"
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute dynamic CTKG construction for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters
                - version_history: Pre-fetched version history (CTKG format)
                - raw_version_data: Raw cthist data with changes pre-computed
                - structure_text: Whether to LLM-parse text fields
            
        Returns:
            TaskResult with version nodes, change relations, and triples
        """
        try:
            nct_id = trial_data.nct_id
            
            # Initialize LLM-dependent components if available
            if self.llm_client or self.api_key:
                self.text_structurer = TextFieldStructurer(self.call_llm)
                self.termination_normalizer = TerminationNormalizer(self.call_llm)
            else:
                self.text_structurer = TextFieldStructurer()
                self.termination_normalizer = TerminationNormalizer()
            
            # Get version history (prefer pre-fetched)
            versions = kwargs.get('version_history', [])
            raw_version_data = kwargs.get('raw_version_data', {})
            
            if not versions:
                versions = self._fetch_version_history(nct_id)
            
            if not versions:
                return self._create_success_result({
                    'version_nodes': [],
                    'change_relations': [],
                    'triples': [],
                    'message': 'No version history available'
                })
            
            # If only single version, still create a version node
            if len(versions) < 2:
                single_node = self._create_version_node(nct_id, versions[0], 1)
                self._tag_version_fields_first(single_node)
                return self._create_success_result({
                    'version_nodes': [single_node.to_dict()],
                    'change_relations': [],
                    'triples': self._generate_triples(nct_id, [single_node], []),
                    'total_versions': 1,
                    'message': 'Single version - no changes to track'
                })
            
            # Create version nodes
            version_nodes = []
            for i, version in enumerate(versions):
                node = self._create_version_node(nct_id, version, i + 1)
                
                # Structure text fields if requested
                if kwargs.get('structure_text', True) and self.text_structurer:
                    node = self._add_structured_text_fields(node, version)
                
                version_nodes.append(node)
            
            # Tag fields with change status for each version
            self._tag_all_version_fields(version_nodes, versions)
            
            # Compute changes between adjacent versions
            change_relations = []
            for i in range(len(versions) - 1):
                change_rel = self.diff_engine.compute_diff(
                    versions[i], 
                    versions[i + 1],
                    nct_id,
                    i + 1,
                    i + 2
                )
                
                # Add entity-level changes for text fields
                if self.text_structurer:
                    self._add_entity_level_changes(
                        change_rel, 
                        version_nodes[i], 
                        version_nodes[i + 1]
                    )
                
                change_relations.append(change_rel)
            
            # Normalize termination reason if applicable
            termination_info = self._process_termination(versions[-1])
            
            # Generate triples for knowledge graph
            triples = self._generate_triples(nct_id, version_nodes, change_relations)
            
            # Include raw cthist changes if available
            cthist_changes = raw_version_data.get('changes', []) if raw_version_data else []
            
            # Generate changes table (tabular summary of all changes)
            changes_table = self.generate_changes_table(version_nodes)
            
            return self._create_success_result({
                'version_nodes': [n.to_dict() for n in version_nodes],
                'change_relations': [cr.to_dict() for cr in change_relations],
                'cthist_changes': cthist_changes,  # Raw changes from R cthist
                'changes_table': changes_table,  # Tabular summary of all changes
                'triples': triples,
                'total_versions': len(versions),
                'first_version_date': raw_version_data.get('first_version_date') if raw_version_data else None,
                'last_version_date': raw_version_data.get('last_version_date') if raw_version_data else None,
                'termination_info': termination_info.to_dict() if termination_info else None,
                'summary': {
                    'versions_analyzed': len(versions),
                    'total_changes': sum(cr.summary['total_changes'] for cr in change_relations),
                    'cthist_change_count': len(cthist_changes),
                    'changes_table_rows': len(changes_table),
                    'fields_with_changes': len(set(
                        fc.field_name 
                        for cr in change_relations 
                        for fc in cr.field_changes
                    ))
                }
            })
            
        except Exception as e:
            logger.error(f"Task 7 failed: {e}", exc_info=True)
            return self._create_error_result([str(e)])
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """Prepare trial data for version tracking."""
        status = getattr(trial_data, 'status', {})
        
        return {
            'nct_id': trial_data.nct_id,
            'current_status': status.get('overallStatus'),
            'why_stopped': status.get('whyStopped'),
            'has_results': trial_data.has_results
        }
    
    def _fetch_version_history(self, nct_id: str) -> List[Dict]:
        """
        Fetch version history using R cthist package or CT.gov API.
        
        The R cthist package provides more reliable access to version history
        compared to the CT.gov API.
        """
        try:
            # First try using R cthist package (more reliable)
            from ..data_loader.cthist_downloader import get_version_history
            versions = get_version_history(nct_id, use_r=True)
            if versions:
                logger.info(f"Successfully fetched {len(versions)} versions for {nct_id}")
                return versions
        except ImportError:
            logger.warning("cthist_downloader not available, trying CT.gov API")
        except Exception as e:
            logger.warning(f"cthist_downloader failed: {e}")
        
        # Fallback to CT.gov API
        try:
            from ..data_loader.ctgov_api import get_client
            client = get_client()
            versions = client.get_all_study_versions_full(nct_id)
            if versions:
                logger.info(f"Fetched {len(versions)} versions via CT.gov API for {nct_id}")
            return versions
        except Exception as e:
            logger.warning(f"Could not fetch version history: {e}")
            return []
    
    def _create_version_node(
        self, 
        nct_id: str, 
        version_data: Dict, 
        version_number: int
    ) -> VersionNode:
        """Create a version node with tracked fields."""
        # Get version date from metadata or data
        version_date = None
        if '_version_metadata' in version_data:
            version_date = version_data['_version_metadata'].get('version_date')
        
        # Extract all tracked fields
        tracked_fields = self.field_extractor.extract_all_fields(version_data)
        
        return VersionNode(
            node_id=f"{nct_id}_v{version_number}",
            nct_id=nct_id,
            version_number=version_number,
            version_date=version_date,
            tracked_fields=tracked_fields,
            metadata=version_data.get('_version_metadata', {})
        )
    
    def _tag_all_version_fields(
        self, 
        version_nodes: List[VersionNode], 
        versions: List[Dict]
    ) -> None:
        """
        Tag all fields in all version nodes with their change status.
        
        For each field in each version:
        - First version: status = FIRST_VERSION
        - Subsequent versions: status = ADDED/REMOVED/MODIFIED/UNCHANGED
        
        Also tags structured_text_fields (eligibility, outcomes) with item-level status.
        
        Args:
            version_nodes: List of VersionNode objects
            versions: Raw version data for comparison
        """
        for i, node in enumerate(version_nodes):
            if i == 0:
                # First version - all fields are FIRST_VERSION
                self._tag_version_fields_first(node)
                # Tag structured text fields for first version
                self._tag_structured_text_fields_first(node)
            else:
                # Subsequent versions - compare with previous
                prev_node = version_nodes[i - 1]
                self._tag_version_fields_diff(node, prev_node, versions[i], versions[i - 1])
                # Tag structured text fields with comparison
                self._tag_structured_text_fields_diff(node, prev_node)
    
    def _tag_structured_text_fields_first(self, node: VersionNode) -> None:
        """Tag all structured text fields in the first version as FIRST_VERSION."""
        if not self.text_structurer:
            return
        
        # Tag eligibility
        eligibility = node.structured_text_fields.get('eligibility', {})
        if eligibility:
            node.tagged_structured_text_fields['eligibility'] = \
                self.text_structurer.tag_structured_eligibility(eligibility, None)
        
        # Tag primary outcomes
        primary_outcomes = node.structured_text_fields.get('primaryOutcomes', [])
        if primary_outcomes:
            node.tagged_structured_text_fields['primaryOutcomes'] = \
                self.text_structurer.tag_structured_outcomes(primary_outcomes, None)
        
        # Tag secondary outcomes
        secondary_outcomes = node.structured_text_fields.get('secondaryOutcomes', [])
        if secondary_outcomes:
            node.tagged_structured_text_fields['secondaryOutcomes'] = \
                self.text_structurer.tag_structured_outcomes(secondary_outcomes, None)
    
    def _tag_structured_text_fields_diff(
        self, 
        current_node: VersionNode, 
        prev_node: VersionNode
    ) -> None:
        """Tag structured text fields with comparison to previous version."""
        if not self.text_structurer:
            return
        
        # Tag eligibility
        curr_elig = current_node.structured_text_fields.get('eligibility', {})
        prev_elig = prev_node.structured_text_fields.get('eligibility', {})
        if curr_elig:
            current_node.tagged_structured_text_fields['eligibility'] = \
                self.text_structurer.tag_structured_eligibility(curr_elig, prev_elig if prev_elig else None)
        
        # Tag primary outcomes
        curr_primary = current_node.structured_text_fields.get('primaryOutcomes', [])
        prev_primary = prev_node.structured_text_fields.get('primaryOutcomes', [])
        if curr_primary:
            current_node.tagged_structured_text_fields['primaryOutcomes'] = \
                self.text_structurer.tag_structured_outcomes(curr_primary, prev_primary if prev_primary else None)
        
        # Tag secondary outcomes
        curr_secondary = current_node.structured_text_fields.get('secondaryOutcomes', [])
        prev_secondary = prev_node.structured_text_fields.get('secondaryOutcomes', [])
        if curr_secondary:
            current_node.tagged_structured_text_fields['secondaryOutcomes'] = \
                self.text_structurer.tag_structured_outcomes(curr_secondary, prev_secondary if prev_secondary else None)
    
    def _tag_version_fields_first(self, node: VersionNode) -> None:
        """Tag all fields in the first version as FIRST_VERSION."""
        for field_name in TRACKED_FIELDS:
            value = node.tracked_fields.get(field_name)
            category = self.field_extractor.get_field_category(field_name)
            
            # For list fields, tag each entity as FIRST_VERSION
            entity_changes = []
            if self.field_extractor.is_list_field(field_name) and isinstance(value, list):
                for item in value:
                    entity_key = self._get_entity_key(item, field_name)
                    entity_changes.append(TaggedEntity(
                        value=item,
                        status=ChangeType.FIRST_VERSION,
                        entity_key=entity_key
                    ))
            
            node.tagged_fields[field_name] = TaggedField(
                field_name=field_name,
                value=value,
                status=ChangeType.FIRST_VERSION,
                previous_value=None,
                category=category,
                entity_changes=entity_changes
            )
    
    def _tag_version_fields_diff(
        self, 
        current_node: VersionNode, 
        prev_node: VersionNode,
        current_data: Dict,
        prev_data: Dict
    ) -> None:
        """
        Tag all fields in a version with their change status relative to previous version.
        
        For list fields, performs entity-level comparison to tag each item as
        added, removed, modified, or unchanged.
        
        Args:
            current_node: Current version node to tag
            prev_node: Previous version node for comparison
            current_data: Raw current version data
            prev_data: Raw previous version data
        """
        for field_name in TRACKED_FIELDS:
            current_value = current_node.tracked_fields.get(field_name)
            prev_value = prev_node.tracked_fields.get(field_name)
            
            # Use diff engine's classification logic
            change_type = self.diff_engine._classify_change(prev_value, current_value)
            category = self.field_extractor.get_field_category(field_name)
            
            # Compute entity-level changes for list fields
            entity_changes = []
            if self.field_extractor.is_list_field(field_name):
                entity_changes = self._compute_entity_level_changes(
                    prev_value, current_value, field_name
                )
            
            # Create tagged field
            tagged_field = TaggedField(
                field_name=field_name,
                value=current_value,
                status=change_type,
                category=category,
                entity_changes=entity_changes
            )
            
            # Add previous value for MODIFIED fields
            if change_type == ChangeType.MODIFIED:
                tagged_field.previous_value = prev_value
            
            current_node.tagged_fields[field_name] = tagged_field
    
    def _compute_entity_level_changes(
        self,
        prev_list: Optional[List],
        curr_list: Optional[List],
        field_name: str
    ) -> List[TaggedEntity]:
        """
        Compute entity-level changes between two lists.
        
        Each entity in the current list is tagged with its change status:
        - added: Entity exists in current but not in previous
        - removed: Entity existed in previous but not in current
        - modified: Entity exists in both but with changes
        - unchanged: Entity is identical in both versions
        
        Args:
            prev_list: Previous version's list
            curr_list: Current version's list
            field_name: Name of the field for key extraction
            
        Returns:
            List of TaggedEntity with change status for each item
        """
        if prev_list is None:
            prev_list = []
        if curr_list is None:
            curr_list = []
        
        entity_changes = []
        
        # Build lookup maps by entity key
        prev_by_key = {}
        for item in prev_list:
            key = self._get_entity_key(item, field_name)
            prev_by_key[key] = item
        
        curr_by_key = {}
        for item in curr_list:
            key = self._get_entity_key(item, field_name)
            curr_by_key[key] = item
        
        prev_keys = set(prev_by_key.keys())
        curr_keys = set(curr_by_key.keys())
        
        # Process current items
        for item in curr_list:
            key = self._get_entity_key(item, field_name)
            
            if key not in prev_keys:
                # Added
                entity_changes.append(TaggedEntity(
                    value=item,
                    status=ChangeType.ADDED,
                    entity_key=key
                ))
            elif key in prev_keys:
                prev_item = prev_by_key[key]
                if self._entities_equal(prev_item, item):
                    # Unchanged
                    entity_changes.append(TaggedEntity(
                        value=item,
                        status=ChangeType.UNCHANGED,
                        entity_key=key
                    ))
                else:
                    # Modified
                    entity_changes.append(TaggedEntity(
                        value=item,
                        status=ChangeType.MODIFIED,
                        entity_key=key,
                        previous_value=prev_item
                    ))
        
        # Add removed items (they exist in prev but not in curr)
        for key in prev_keys - curr_keys:
            entity_changes.append(TaggedEntity(
                value=prev_by_key[key],
                status=ChangeType.REMOVED,
                entity_key=key
            ))
        
        return entity_changes
    
    def _get_entity_key(self, item: Any, field_name: str) -> str:
        """
        Generate a unique key for an entity based on its identifying fields.
        
        Different field types have different key fields:
        - locations: city + state + country
        - interventions: name
        - outcomes: measure
        - contacts: name
        """
        if not isinstance(item, dict):
            return str(item)
        
        if field_name == 'locations':
            city = item.get('city', '')
            state = item.get('state', '')
            country = item.get('country', '')
            facility = item.get('facility', '')
            return f"{facility or city}, {state}, {country}".strip(', ')
        
        elif field_name in ['interventions', 'armGroups']:
            return item.get('name', '') or item.get('label', '') or str(item)[:50]
        
        elif field_name in ['primaryOutcomes', 'secondaryOutcomes']:
            return item.get('measure', '')[:80] or str(item)[:50]
        
        elif field_name in ['centralContacts', 'overallContacts']:
            name = item.get('name', '')
            role = item.get('role', '')
            return f"{name} ({role})" if role else name
        
        elif field_name == 'collaborators':
            return item.get('name', '') or str(item)[:50]
        
        else:
            # Generic: try common fields
            for key in ['name', 'title', 'measure', 'city', 'id']:
                if key in item:
                    return str(item[key])[:80]
            return json.dumps(item, sort_keys=True)[:80]
    
    def _entities_equal(self, item1: Any, item2: Any) -> bool:
        """Check if two entities are equal."""
        if type(item1) != type(item2):
            return False
        
        if isinstance(item1, dict):
            # Compare dictionaries
            if set(item1.keys()) != set(item2.keys()):
                return False
            return all(
                self._entities_equal(item1.get(k), item2.get(k))
                for k in item1.keys()
            )
        elif isinstance(item1, list):
            if len(item1) != len(item2):
                return False
            return all(self._entities_equal(a, b) for a, b in zip(item1, item2))
        else:
            return item1 == item2
    
    def _add_structured_text_fields(
        self, 
        node: VersionNode, 
        version_data: Dict
    ) -> VersionNode:
        """Add LLM-structured text fields to version node."""
        # Structure eligibility criteria
        eligibility_text = self.field_extractor.extract_field(
            version_data, 'eligibilityCriteria'
        )
        if eligibility_text:
            node.structured_text_fields['eligibility'] = \
                self.text_structurer.structure_eligibility(eligibility_text)
        
        # Structure outcomes
        primary_outcomes = self.field_extractor.extract_field(
            version_data, 'primaryOutcomes'
        )
        if primary_outcomes:
            node.structured_text_fields['primaryOutcomes'] = \
                self.text_structurer.structure_outcomes(primary_outcomes)
        
        secondary_outcomes = self.field_extractor.extract_field(
            version_data, 'secondaryOutcomes'
        )
        if secondary_outcomes:
            node.structured_text_fields['secondaryOutcomes'] = \
                self.text_structurer.structure_outcomes(secondary_outcomes)
        
        return node
    
    def _add_entity_level_changes(
        self,
        change_rel: VersionChangeRelation,
        v1_node: VersionNode,
        v2_node: VersionNode
    ) -> None:
        """Add entity-level changes for text-intensive fields."""
        for fc in change_rel.field_changes:
            if fc.field_name == 'eligibilityCriteria':
                v1_elig = v1_node.structured_text_fields.get('eligibility', {})
                v2_elig = v2_node.structured_text_fields.get('eligibility', {})
                
                if v1_elig and v2_elig:
                    fc.entity_level_changes = [
                        self.text_structurer.compare_structured_eligibility(
                            v1_elig, v2_elig
                        )
                    ]
            
            elif fc.field_name in ['primaryOutcomes', 'secondaryOutcomes']:
                v1_out = v1_node.structured_text_fields.get(fc.field_name, [])
                v2_out = v2_node.structured_text_fields.get(fc.field_name, [])
                
                if v1_out and v2_out:
                    fc.entity_level_changes = [
                        self.text_structurer.compare_structured_outcomes(
                            v1_out, v2_out
                        )
                    ]
    
    def _process_termination(self, final_version: Dict) -> Optional[TerminationInfo]:
        """Process and normalize termination reason."""
        why_stopped = self.field_extractor.extract_field(final_version, 'whyStopped')
        status = self.field_extractor.extract_field(final_version, 'overallStatus')
        
        if not why_stopped and status not in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            return None
        
        return self.termination_normalizer.normalize(why_stopped, status)
    
    def _generate_triples(
        self, 
        nct_id: str,
        version_nodes: List[VersionNode],
        change_relations: List[VersionChangeRelation]
    ) -> List[Dict]:
        """
        Generate knowledge graph triples for dynamic CTKG.
        
        Triple types:
        1. Version node triples with change status tags
        2. Version chain triples (v1 -> nextVersion -> v2)
        3. Change relation triples (change -> hasType -> type)
        4. Canonical link triples (version -> isVersionOf -> NCTID)
        
        All triples include a 'change_status' attribute indicating:
        - first_version: First version, no comparison
        - added: New in this version
        - removed: Was in previous version, not in this version
        - modified: Changed from previous version
        - unchanged: Same as previous version
        """
        triples = []
        
        # 1. Version node triples with change status
        for node in version_nodes:
            # Link to canonical NCTID
            triples.append({
                'head': node.node_id,
                'relation': 'isVersionOf',
                'tail': node.nct_id,
                'head_type': 'TrialVersion',
                'tail_type': 'Trial'
            })
            
            # Version date
            if node.version_date:
                triples.append({
                    'head': node.node_id,
                    'relation': 'hasVersionDate',
                    'tail': node.version_date,
                    'head_type': 'TrialVersion',
                    'tail_type': 'Date'
                })
            
            # Use tagged_fields to get status for each field
            for field_name, tagged_field in node.tagged_fields.items():
                value = tagged_field.value
                status = tagged_field.status.value
                
                # Handle scalar fields
                if value is not None and not isinstance(value, (list, dict)):
                    triples.append({
                        'head': node.node_id,
                        'relation': f'has{field_name[0].upper()}{field_name[1:]}',
                        'tail': str(value),
                        'head_type': 'TrialVersion',
                        'tail_type': self._get_tail_type(field_name),
                        'attributes': {
                            'change_status': status,
                            'previous_value': str(tagged_field.previous_value) if tagged_field.previous_value else None
                        }
                    })
                
                # Handle list fields with entity-level changes
                elif tagged_field.entity_changes:
                    for entity in tagged_field.entity_changes:
                        entity_value = self._format_entity_value(entity.value, field_name)
                        entity_status = entity.status.value
                        
                        # Skip unchanged entities to reduce noise (optional)
                        # if entity_status == 'unchanged':
                        #     continue
                        
                        triples.append({
                            'head': node.node_id,
                            'relation': f'has{field_name[0].upper()}{field_name[1:]}',
                            'tail': entity_value,
                            'head_type': 'TrialVersion',
                            'tail_type': self._get_entity_type(field_name),
                            'attributes': {
                                'change_status': entity_status,
                                'entity_key': entity.entity_key,
                                'previous_value': self._format_entity_value(entity.previous_value, field_name) if entity.previous_value else None
                            }
                        })
            
            # Add triples for tagged_structured_text_fields
            for field_name, tagged_stf in node.tagged_structured_text_fields.items():
                if field_name == 'eligibility':
                    for criteria_type in ['inclusion', 'exclusion']:
                        items = tagged_stf.get(criteria_type, [])
                        for item in items:
                            status = item.get('status', 'unchanged')
                            value = item.get('value', '')
                            prev_value = item.get('previous_value', '')
                            
                            triples.append({
                                'head': node.node_id,
                                'relation': f'hasEligibility_{criteria_type.capitalize()}',
                                'tail': self._truncate(value, 200),
                                'head_type': 'TrialVersion',
                                'tail_type': 'EligibilityCriterion',
                                'attributes': {
                                    'change_status': status,
                                    'previous_value': self._truncate(prev_value, 200) if prev_value else None
                                }
                            })
                
                elif field_name in ['primaryOutcomes', 'secondaryOutcomes']:
                    if isinstance(tagged_stf, list):
                        for outcome in tagged_stf:
                            status = outcome.get('status', 'unchanged')
                            measure = outcome.get('measure', '')
                            prev = outcome.get('previous_value', {})
                            prev_measure = prev.get('measure', '') if isinstance(prev, dict) else ''
                            
                            triples.append({
                                'head': node.node_id,
                                'relation': f'has{field_name[0].upper()}{field_name[1:]}',
                                'tail': self._truncate(measure, 200),
                                'head_type': 'TrialVersion',
                                'tail_type': 'Outcome',
                                'attributes': {
                                    'change_status': status,
                                    'timeFrame': outcome.get('timeFrame', ''),
                                    'previous_value': self._truncate(prev_measure, 200) if prev_measure else None
                                }
                            })
        
        # 2. Version chain triples
        for i in range(len(version_nodes) - 1):
            triples.append({
                'head': version_nodes[i].node_id,
                'relation': 'nextVersion',
                'tail': version_nodes[i + 1].node_id,
                'head_type': 'TrialVersion',
                'tail_type': 'TrialVersion'
            })
        
        # 3. Change relation triples
        for change_rel in change_relations:
            change_id = f"{change_rel.from_version}_to_{change_rel.to_version}"
            
            # Link change to versions
            triples.append({
                'head': change_id,
                'relation': 'fromVersion',
                'tail': change_rel.from_version,
                'head_type': 'VersionChange',
                'tail_type': 'TrialVersion'
            })
            triples.append({
                'head': change_id,
                'relation': 'toVersion',
                'tail': change_rel.to_version,
                'head_type': 'VersionChange',
                'tail_type': 'TrialVersion'
            })
            
            # Field changes
            for fc in change_rel.field_changes:
                triples.append({
                    'head': change_id,
                    'relation': f'{fc.change_type.value}Field',
                    'tail': fc.field_name,
                    'head_type': 'VersionChange',
                    'tail_type': 'Field',
                    'attributes': {
                        'change_type': fc.change_type.value,
                        'category': fc.category.value if fc.category else None
                    }
                })
                
                # Entity-level changes for text fields
                for entity_change in fc.entity_level_changes:
                    if isinstance(entity_change, dict):
                        for change_type, items in entity_change.items():
                            if isinstance(items, list):
                                for item in items:
                                    if item:
                                        triples.append({
                                            'head': change_id,
                                            'relation': f'{change_type}Entity',
                                            'tail': str(item)[:200],
                                            'head_type': 'VersionChange',
                                            'tail_type': 'Entity',
                                            'attributes': {'field': fc.field_name}
                                        })
        
        return triples
    
    def _get_tail_type(self, field: str) -> str:
        """Get the tail type for a field triple."""
        type_map = {
            'overallStatus': 'Status',
            'startDate': 'Date',
            'completionDate': 'Date',
            'primaryCompletionDate': 'Date',
            'enrollmentCount': 'Number',
            'enrollmentType': 'EnrollmentType',
            'phase': 'Phase',
            'studyType': 'StudyType',
            'whyStopped': 'TerminationReason'
        }
        return type_map.get(field, 'Value')
    
    def _get_entity_type(self, field: str) -> str:
        """Get the entity type for list field items."""
        entity_type_map = {
            'locations': 'Location',
            'centralContacts': 'Contact',
            'interventions': 'Intervention',
            'armGroups': 'ArmGroup',
            'primaryOutcomes': 'Outcome',
            'secondaryOutcomes': 'Outcome',
            'conditions': 'Condition',
            'collaborators': 'Sponsor'
        }
        return entity_type_map.get(field, 'Entity')
    
    def _format_entity_value(self, entity: Any, field: str) -> str:
        """Format an entity value for display in triples."""
        if entity is None:
            return ""
        
        if isinstance(entity, dict):
            # Location: city, state, country
            if 'city' in entity:
                parts = [entity.get('city', ''), entity.get('state', ''), entity.get('country', '')]
                return ', '.join(p for p in parts if p)
            # Contact: name
            elif 'name' in entity:
                return entity.get('name', '')
            # Intervention: name or type
            elif 'interventionName' in entity:
                return entity.get('interventionName', '')
            # Arm group: label
            elif 'label' in entity:
                return entity.get('label', '')
            # Generic dict
            else:
                return str(entity)[:150]
        elif isinstance(entity, str):
            return entity[:150]
        else:
            return str(entity)[:150]
    
    def generate_changes_table(
        self, 
        version_nodes: List[VersionNode]
    ) -> List[Dict]:
        """
        Generate a tabular representation of all changes across versions.
        
        Each row represents a single change:
        - nct_id: Trial ID
        - version: Version number (e.g., "v1 → v2")
        - field: Field name that changed
        - old_value: Previous value
        - new_value: New value
        - change_type: Type of change (added, removed, modified)
        
        Excludes unchanged fields.
        
        Args:
            version_nodes: List of VersionNode objects
            
        Returns:
            List of change records as dictionaries
        """
        changes_table = []
        
        for i, node in enumerate(version_nodes):
            if i == 0:
                continue  # Skip first version (no changes to compare)
            
            nct_id = node.nct_id
            version_label = f"v{i} → v{i+1}"
            
            # Process tagged_fields (tracked fields)
            for field_name, tagged_field in node.tagged_fields.items():
                if tagged_field.status == ChangeType.UNCHANGED:
                    continue
                if tagged_field.status == ChangeType.FIRST_VERSION:
                    continue
                
                # For list fields with entity-level changes
                if tagged_field.entity_changes:
                    for entity in tagged_field.entity_changes:
                        if entity.status == ChangeType.UNCHANGED:
                            continue
                        if entity.status == ChangeType.FIRST_VERSION:
                            continue
                        
                        old_val = self._format_value(entity.previous_value) if entity.previous_value else ""
                        new_val = self._format_value(entity.value) if entity.status != ChangeType.REMOVED else ""
                        
                        if entity.status == ChangeType.REMOVED:
                            old_val = self._format_value(entity.value)
                            new_val = ""
                        
                        # Skip if formatted values are the same (internal change like geoPoint)
                        # These are not meaningful changes from a user perspective
                        if entity.status == ChangeType.MODIFIED and old_val == new_val:
                            continue
                        
                        changes_table.append({
                            'nct_id': nct_id,
                            'version': version_label,
                            'field': field_name,
                            'old_value': old_val,
                            'new_value': new_val,
                            'change_type': entity.status.value
                        })
                else:
                    # Scalar field change
                    changes_table.append({
                        'nct_id': nct_id,
                        'version': version_label,
                        'field': field_name,
                        'old_value': self._format_value(tagged_field.previous_value),
                        'new_value': self._format_value(tagged_field.value),
                        'change_type': tagged_field.status.value
                    })
            
            # Process tagged_structured_text_fields
            for field_name, tagged_stf in node.tagged_structured_text_fields.items():
                if field_name == 'eligibility':
                    # Eligibility has inclusion/exclusion
                    for criteria_type in ['inclusion', 'exclusion']:
                        items = tagged_stf.get(criteria_type, [])
                        for item in items:
                            status = item.get('status', '')
                            if status in ['unchanged', 'first_version']:
                                continue
                            
                            value = item.get('value', '')
                            prev_value = item.get('previous_value', '')
                            
                            if status == 'removed':
                                old_val = self._truncate(value, 100)
                                new_val = ""
                            elif status == 'added':
                                old_val = ""
                                new_val = self._truncate(value, 100)
                            else:  # modified
                                old_val = self._truncate(prev_value, 100)
                                new_val = self._truncate(value, 100)
                            
                            changes_table.append({
                                'nct_id': nct_id,
                                'version': version_label,
                                'field': f'eligibility.{criteria_type}',
                                'old_value': old_val,
                                'new_value': new_val,
                                'change_type': status
                            })
                
                elif field_name in ['primaryOutcomes', 'secondaryOutcomes']:
                    # Outcomes list
                    if isinstance(tagged_stf, list):
                        for outcome in tagged_stf:
                            status = outcome.get('status', '')
                            if status in ['unchanged', 'first_version']:
                                continue
                            
                            measure = outcome.get('measure', '')
                            prev = outcome.get('previous_value', {})
                            prev_measure = prev.get('measure', '') if isinstance(prev, dict) else ''
                            
                            if status == 'removed':
                                old_val = self._truncate(measure, 100)
                                new_val = ""
                            elif status == 'added':
                                old_val = ""
                                new_val = self._truncate(measure, 100)
                            else:  # modified
                                old_val = self._truncate(prev_measure, 100)
                                new_val = self._truncate(measure, 100)
                            
                            changes_table.append({
                                'nct_id': nct_id,
                                'version': version_label,
                                'field': field_name,
                                'old_value': old_val,
                                'new_value': new_val,
                                'change_type': status
                            })
        
        return changes_table
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display in the changes table."""
        if value is None:
            return ""
        if isinstance(value, dict):
            # For location-like entities, create readable string
            if 'city' in value:
                parts = [value.get('city', ''), value.get('state', ''), value.get('country', '')]
                return ', '.join(p for p in parts if p)
            elif 'name' in value:
                return value.get('name', '')
            else:
                return str(value)[:100]
        elif isinstance(value, list):
            return f"[{len(value)} items]"
        else:
            return self._truncate(str(value), 100)
    
    def _truncate(self, text: str, max_len: int = 100) -> str:
        """Truncate text to max length with ellipsis."""
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
    
    def export_changes_table_csv(
        self, 
        version_nodes: List[VersionNode],
        output_path: str
    ) -> None:
        """
        Export changes table to CSV file.
        
        Args:
            version_nodes: List of VersionNode objects
            output_path: Path to output CSV file
        """
        import csv
        
        changes = self.generate_changes_table(version_nodes)
        
        if not changes:
            logger.warning("No changes to export")
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['nct_id', 'version', 'field', 'old_value', 'new_value', 'change_type'])
            writer.writeheader()
            writer.writerows(changes)
        
        logger.info(f"Changes table exported to {output_path} ({len(changes)} changes)")

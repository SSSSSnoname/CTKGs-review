"""
Task 1: Standardizing Outcome Measures

Decomposes outcome measures into core indicators and attributes:
- Core measurement concept (e.g., HbA1c, blood pressure)
- Time frame
- Measurement instrument
- Unit
- Numerical constraints
- Population qualifiers

The extraction process identifies:
- Core measurement concepts (atomic, standardized biomedical concepts)
- Measurement tools/instruments
- Value conditions (thresholds, criteria)
- Conditional populations (subgroup qualifiers)

The goal is to produce normalized concepts suitable for knowledge graph entities
that can be mapped to standard ontologies (SNOMED CT, MeSH, UMLS).
"""

from typing import Dict, List, Any, Tuple
import json
import logging
import re

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

OUTCOME_PROMPT_TEMPLATE = """
# Role
You are a Clinical Trial Outcome Normalization Engine.

# Input
You will receive:
(1) outcome_title: a text outcome measure title
(2) structured_fields: a JSON object that may contain timeFrame, unitOfMeasure/units, and other attributes.

# Goal
Extract normalized "core measurement" concepts that are suitable as knowledge graph entities and can be mapped to ontology entities (e.g., SNOMED CT, MeSH, UMLS). Fill supplementary fields ONLY when missing from structured_fields.

# Definitions
Core measurement = a standardized endpoint concept that represents the fundamental outcome being assessed. It can be an event/state/quantity/disease/symptom/lab value.

CRITICAL REQUIREMENTS for core_measurement:
1. **Use original words from outcome_title**: The core_measurement MUST use the exact words/phrases that appear in the outcome_title, unless expanding abbreviations. Do not use synonyms or add words not present in the original title (e.g., if title says "symptom free", use "symptom free" not "asymptomatic"; if title says "pain relief", use "pain relief" not "analgesia").
2. **Atomic concept**: Extract a SINGLE, indivisible concept. Avoid compound phrases that combine multiple concepts.
   - GOOD: "blood pressure", "heart rate", "pain intensity", "tumor size"
   - BAD: "change in blood pressure" (includes temporal concept), "blood pressure reduction" (includes direction), "systolic blood pressure at baseline" (includes qualifiers)
   - If the outcome_title itself is already a standardized clinical endpoint term  (e.g., "Overall Survival", "Progression-Free Survival", "Overall Response Rate"), retain the full endpoint term as the core_measurement after expanding abbreviations.Do NOT reduce it to a more generic underlying concept (e.g., do NOT reduce "Overall Response Rate" to "response").
3. **MUST NOT include**:
   - Statistical framing: percentage, number of participants, incidence, rate, proportion, count
   - Time information: baseline, follow-up, post-treatment, duration, time points
   - Study design phrases: randomization, change from baseline, improvement
   - Directional qualifiers: increase, decrease, reduction, improvement, worsening
   - Population qualifiers: specific patient groups, subgroups
   - Measurement context: "at rest", "during exercise", "after meal"

# Instructions
1) Extract core_measurement(s) from outcome_title:
   - Identify the fundamental biomedical concept being measured
   - Strip away all modifiers, qualifiers, and contextual information
   
   Examples of extraction:
   - "Change in systolic blood pressure from baseline" → "systolic blood pressure"
   - "Percentage of participants with response" → "response"
   - "Time to disease progression" → "disease progression"
   - "Number of adverse events" → "adverse event"
   - "Quality of life score" → "quality of life"
   
2) For supplementary fields:
   - measurement_tool: only output a tool/instrument if explicitly mentioned; otherwise "Not Applicable".
   - value_condition: only output numeric thresholds/criteria if explicitly present; otherwise "Not Applicable".
   - conditional_population: only output subgroup qualifiers if explicitly present in title AND not merely "participants"; otherwise "Not Applicable".

3) Expand all abbreviations to full forms (e.g., "BP" → "blood pressure", "HR" → "heart rate", "DLT" → "dose limiting toxicity").

4) If an outcome title contains multiple distinct measures, extract them separately:
   - Each distinct measure should be a separate object in the output array
   - Example: "Blood pressure and heart rate" → two separate objects: one for "blood pressure", one for "heart rate"
   - Only split when measures are clearly distinct concepts (connected by "and", "or", commas, etc.)
   - If concepts are closely related or one modifies the other, keep them together (e.g., "systolic blood pressure" is one concept, not "systolic" + "blood pressure")

5) Output JSON array. Each object must have these keys:
   core_measurement, measurement_tool, value_condition, conditional_population

# Output Format
Return ONLY valid JSON array. Each object in the array must contain all required keys:
[
  {{
    "core_measurement": "normalized core measurement concept",
    "measurement_tool": "tool/instrument name or Not Applicable",
    "value_condition": "numeric threshold/criteria or Not Applicable",
    "conditional_population": "subgroup qualifier or Not Applicable"
  }}
]

**Example Input1:**
Outcome Measure: 'Change of neurological morbidity in young adults with sickle cell anemia assesed by neurological examinations, Magnetic Resonance Angiography, and Magnetic Resonance Imaging'
Existing Attributes: {{"paramType": "Mean", "timeFrame": "12 weeks", "unitOfMeasure": null, "units": null}}

**Example Output1:**
[
  {{
    "core_measurement": "neurological morbidity",
    "measurement_tool": ["neurological examinations", "Magnetic Resonance Angiography", "Magnetic Resonance Imaging"],
    "value_condition": "Not Applicable",
    "conditional_population": "young adults with sickle cell anemia"
  }}
]
Note: "neurological morbidity" is a standardized concept that can be mapped to ontology entities. The temporal "Change" and measurement context are removed.

**Example Input2:**
Outcome Measure: 'MTD of veliparib in combination with carboplatin and paclitaxel, determined according to incidence of DLT as graded using the National Cancer Institute Common Terminology Criteria for Adverse Events (NCI CTCAE) version (v) 4.0'
Existing Attributes: {{"paramType": null, "timeFrame": "21 days", "unitOfMeasure": null, "units": null}}

**Example Output2:**
[
  {{
    "core_measurement": "maximum tolerated dose",
    "measurement_tool": "National Cancer Institute Common Terminology Criteria for Adverse Events version 4.0",
    "value_condition": "Not Applicable",
    "conditional_population": "Not Applicable"
  }},
  {{
    "core_measurement": "dose limiting toxicity",
    "measurement_tool": "National Cancer Institute Common Terminology Criteria for Adverse Events version 4.0",
    "value_condition": "Not Applicable",
    "conditional_population": "Not Applicable"
  }}
]
Note: Both "maximum tolerated dose" and "dose limiting toxicity" are atomic, standardized concepts. The drug names and combination information are removed from core_measurement.

# Input Data

Outcome Measure: '{outcome_title}'
Existing Attributes: {existing_attributes}
"""


# =============================================================================
# PROMPT AND PARSING FUNCTIONS
# =============================================================================

def get_outcome_prompt(outcome_data: Dict) -> str:
    """
    Generate the complete outcome extraction prompt.
    
    Args:
        outcome_data: Dictionary containing:
            - outcome_title: Outcome measure title
            - existing_attributes: Dict with paramType, timeFrame, unitOfMeasure, units
    
    Returns:
        Complete prompt string ready for LLM input
    """
    outcome_title = str(outcome_data.get('outcome_title', '') or '')
    existing_attrs = outcome_data.get('existing_attributes', {})
    
    attrs_dict = {
        "paramType": existing_attrs.get('paramType'),
        "timeFrame": existing_attrs.get('timeFrame'),
        "unitOfMeasure": existing_attrs.get('unitOfMeasure'),
        "units": existing_attrs.get('units')
    }
    existing_attributes_str = json.dumps(attrs_dict, ensure_ascii=False)
    
    # Use string replacement to avoid brace conflicts
    prompt = OUTCOME_PROMPT_TEMPLATE.replace('{outcome_title}', '___OUTCOME_TITLE___')
    prompt = prompt.replace('{existing_attributes}', '___EXISTING_ATTRIBUTES___')
    
    prompt = prompt.replace('___OUTCOME_TITLE___', outcome_title)
    prompt = prompt.replace('___EXISTING_ATTRIBUTES___', existing_attributes_str)
    
    return prompt


def _clean_key(input_string: str) -> str:
    """Normalize key names by removing symbols, spaces and converting to lowercase."""
    return re.sub(r'[^a-zA-Z0-9]', '', input_string).lower()


def validate_outcome_format(parsed_result: List) -> Tuple[bool, str]:
    """
    Validate that the parsed outcome result has the required format.
    
    Args:
        parsed_result: Parsed JSON result (should be a list)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(parsed_result, list):
        return False, "Result is not a list"
    
    required_keys = {
        _clean_key("core_measurement"),
        _clean_key("measurement_tool"),
        _clean_key("value_condition"),
        _clean_key("conditional_population")
    }
    
    for item in parsed_result:
        if not isinstance(item, dict):
            return False, "Element in result is not a dictionary"
        item_keys = {_clean_key(key) for key in item.keys()}
        if not required_keys.issubset(item_keys):
            missing = required_keys - item_keys
            return False, f"Missing required keys: {missing}"
    
    return True, "Valid format"


def parse_outcome_response(response_content: str) -> List:
    """
    Parse LLM response to extract outcome information.
    
    Args:
        response_content: Raw response content from LLM
    
    Returns:
        Parsed list of outcome dictionaries, or dict with error info if parsing fails
    """
    json_match = re.search(r'(\[.*\]|\{.*\})', response_content, re.DOTALL)
    if json_match:
        try:
            parsed_result = json.loads(json_match.group())
            # If result is an object, try to extract array
            if isinstance(parsed_result, dict):
                for key in ['outcomes', 'results', 'data']:
                    if key in parsed_result and isinstance(parsed_result[key], list):
                        return parsed_result[key]
                # Wrap single object in list
                return [parsed_result]
            return parsed_result
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_content": response_content[:500]
            }
    return {
        "error": "No JSON found in response",
        "raw_content": response_content[:500]
    }


class Task1OutcomeStandardization(BaseTaskHandler):
    """
    Task 1: Standardizing Outcome Measures
    
    Uses a "core indicator with attributes" schema to decompose each outcome
    into its primary indicator and contextual attributes.
    """
    
    @property
    def task_name(self) -> str:
        return "task1_outcome_standardization"

    def _to_flat_records(self, outcome_input: Dict, parsed: Any) -> List[Dict]:
        """
        Convert LLM parsed output to final flat schema:
        {
          "original_title": str,
          "core_measurement": str,
          "attributes": {...}
        }
        """
        if not isinstance(parsed, list):
            return []

        records: List[Dict] = []
        existing_attrs = outcome_input.get('existing_attributes', {}) or {}
        # Keep a stable attribute schema across results/protocol sources.
        normalized_existing_attrs = {
            'paramType': existing_attrs.get('paramType'),
            'timeFrame': existing_attrs.get('timeFrame'),
            'unitOfMeasure': existing_attrs.get('unitOfMeasure'),
            'units': existing_attrs.get('units'),
            'description': existing_attrs.get('description'),
        }

        for item in parsed:
            if not isinstance(item, dict):
                continue
            core = item.get('core_measurement')
            if not core:
                continue

            attrs = dict(normalized_existing_attrs)
            attrs.update({
                'outcome_type': outcome_input.get('outcome_type'),
                'measurement_tool': item.get('measurement_tool', 'Not Applicable'),
                'value_condition': item.get('value_condition', 'Not Applicable'),
                'conditional_population': item.get('conditional_population', 'Not Applicable')
            })

            records.append({
                'original_title': outcome_input.get('outcome_title', ''),
                'core_measurement': core,
                'attributes': attrs
            })

        return records
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute outcome standardization for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with standardized outcomes
        """
        try:
            # Prepare input data
            inputs = self.prepare_input(trial_data)
            
            if not inputs.get('outcomes'):
                return self._create_success_result({
                    'standardized_outcomes': [],
                    'message': 'No outcomes found in trial data'
                })
            
            # Process each outcome
            standardized_outcomes = []
            errors = []
            
            for outcome_input in inputs['outcomes']:
                try:
                    # Generate prompt
                    prompt = get_outcome_prompt(outcome_input)
                    
                    # Call LLM
                    response = self.call_llm(prompt)
                    
                    # Parse response
                    parsed = parse_outcome_response(response)
                    
                    standardized_outcomes.extend(
                        self._to_flat_records(outcome_input, parsed)
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process outcome: {e}")
                    errors.append(str(e))
            
            if errors and not standardized_outcomes:
                return self._create_error_result(errors)
            
            return self._create_success_result({
                'standardized_outcomes': standardized_outcomes,
                'total_outcomes': len(inputs['outcomes']),
                'processed_outcomes': len(standardized_outcomes),
                'errors': errors
            })
            
        except Exception as e:
            logger.error(f"Task 1 failed: {e}")
            return self._create_error_result([str(e)])
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare outcome data for standardization.
        
        Extracts outcomes from:
        1. resultsSection.outcomeMeasuresModule (posted results)
        2. protocolSection.outcomesModule (protocol outcomes)
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with 'outcomes' list
        """
        outcomes = []
        
        # Extract from results section (if available)
        results_outcomes = self._extract_from_results(trial_data)
        outcomes.extend(results_outcomes)
        
        # If no results, try protocol outcomes
        if not outcomes:
            protocol_outcomes = self._extract_from_protocol(trial_data)
            outcomes.extend(protocol_outcomes)
        
        return {'outcomes': outcomes}
    
    def _extract_from_results(self, trial_data: Any) -> List[Dict]:
        """Extract outcomes from results section."""
        outcomes = []
        
        results_section = getattr(trial_data, 'results_section', {})
        outcome_measures = results_section.get('outcomeMeasuresModule', {})
        measures = outcome_measures.get('outcomeMeasures', [])
        
        for measure in measures:
            outcome_input = {
                'outcome_title': measure.get('title', ''),
                'outcome_type': measure.get('type', ''),
                'existing_attributes': {
                    'paramType': measure.get('paramType'),
                    'timeFrame': measure.get('timeFrame'),
                    'unitOfMeasure': measure.get('unitOfMeasure'),
                    'units': measure.get('unitOfMeasure'),
                    'description': measure.get('description')
                }
            }
            outcomes.append(outcome_input)
        
        return outcomes
    
    def _extract_from_protocol(self, trial_data: Any) -> List[Dict]:
        """Extract outcomes from protocol section."""
        outcomes = []
        
        outcomes_module = getattr(trial_data, 'outcomes', {})
        
        # Primary outcomes
        for outcome in outcomes_module.get('primaryOutcomes', []):
            outcome_input = {
                'outcome_title': outcome.get('measure', ''),
                'outcome_type': 'PRIMARY',
                'existing_attributes': {
                    'paramType': None,
                    'timeFrame': outcome.get('timeFrame'),
                    'unitOfMeasure': None,
                    'units': None,
                    'description': outcome.get('description')
                }
            }
            outcomes.append(outcome_input)
        
        # Secondary outcomes
        for outcome in outcomes_module.get('secondaryOutcomes', []):
            outcome_input = {
                'outcome_title': outcome.get('measure', ''),
                'outcome_type': 'SECONDARY',
                'existing_attributes': {
                    'paramType': None,
                    'timeFrame': outcome.get('timeFrame'),
                    'unitOfMeasure': None,
                    'units': None,
                    'description': outcome.get('description')
                }
            }
            outcomes.append(outcome_input)
        
        return outcomes

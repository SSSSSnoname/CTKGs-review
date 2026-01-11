"""
Task 3: Structuring Eligibility Criteria

Parses eligibility criteria into structured filters using entity-level extraction:
- Medical conditions (diseases mentioned in criteria)
- Pregnancy requirements (Yes/No/Unknown)
- Medications or therapy (drugs/treatments mentioned)
- Demographic constraints
- Baseline characteristics (where available)

Outputs standardized entities suitable for knowledge graph construction.
"""

from typing import Dict, List, Any
import json
import logging

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# Two-part prompt template for eligibility structuring (optimized for entity extraction)

ELIGIBILITY_PROMPT_PART1 = """
# Role
You are a Clinical Trial Eligibility Criteria Parser specialized in extracting structured entities.

# Task
Summarize the inclusion and exclusion criteria of the following clinical trial using entities to filter specific information. Focus on extracting:

1. **Medical conditions**: The name of diseases/conditions mentioned in the criteria
   - Extract each disease/condition as a separate standardized entity
   - Use the most standard medical terminology (e.g., "Type 2 Diabetes Mellitus" not "T2DM")

2. **Pregnancy**: Whether the trial population has pregnancy requirements
   - "Yes": The criteria explicitly mention that participants MUST be pregnant
   - "No": The criteria explicitly PROHIBIT pregnancy or require contraception
   - "Unknown": No reference to pregnancy in the criteria

3. **Medications or therapy**: Drugs or treatments mentioned in the criteria
   - Include both required and prohibited medications/therapies
   - Use standard drug names (generic names preferred)
   - If combination therapy is mentioned, specify the complete combination

4. **Laboratory values**: Specific test thresholds or biomarker requirements

5. **Demographics**: Age, sex, and other demographic constraints

# Key Instructions
- Ensure each entity appears in its most STANDARD and CONCISE form suitable for a knowledge graph
- Do NOT use abbreviations unless they are the standard form
- If there is no relevant information for a category, output "unknown" or empty list
- Separate inclusion and exclusion criteria clearly

# Input Eligibility Criteria
{eligibility_text}
"""

ELIGIBILITY_PROMPT_PART2 = """
# Output Format
Generate a valid JSON object in the following structure:

{{
  "Inclusion": {{
    "Medical conditions": ["disease 1", "disease 2", ...],
    "Pregnancy": "Yes" | "No" | "Unknown",
    "Medications or therapy": ["drug/therapy 1", "drug/therapy 2", ...],
    "Laboratory values": ["lab criteria 1", "lab criteria 2", ...],
    "Demographics": {{
      "age_min": "minimum age or null",
      "age_max": "maximum age or null", 
      "sex": "male/female/all or null"
    }},
    "Other": ["other criteria 1", ...]
  }},
  "Exclusion": {{
    "Medical conditions": ["disease 1", "disease 2", ...],
    "Pregnancy": "Yes" | "No" | "Unknown",
    "Medications or therapy": ["drug/therapy 1", "drug/therapy 2", ...],
    "Laboratory values": ["lab criteria 1", ...],
    "Demographics": {{}},
    "Other": ["other exclusion criteria 1", ...]
  }},
  "Summary": "Brief one-sentence summary of the target population"
}}

Return ONLY the JSON object, no additional text.
"""


class Task3EligibilityStructuring(BaseTaskHandler):
    """
    Task 3: Structuring Eligibility Criteria
    
    Parses eligibility criteria into structured entity-level constraints
    suitable for knowledge graph construction and cohort modeling.
    
    Key outputs:
    - Medical conditions (standardized disease entities)
    - Pregnancy status (Yes/No/Unknown)
    - Medications or therapy (standardized drug/therapy entities)
    - Laboratory values
    - Demographics
    """
    
    @property
    def task_name(self) -> str:
        return "task3_eligibility_structuring"
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute eligibility structuring for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with structured eligibility criteria as entities
        """
        try:
            # Prepare input data
            inputs = self.prepare_input(trial_data)
            
            if not inputs.get('eligibility_text'):
                return self._create_success_result({
                    'structured_eligibility': self._get_empty_structure(),
                    'message': 'No eligibility criteria found'
                })
            
            # Generate two-part prompt for better entity extraction
            prompt = self._build_full_prompt(inputs['eligibility_text'])
            
            # Call LLM
            response = self.call_llm(prompt)
            
            # Parse response
            structured = self._parse_response(response)
            
            # Validate and normalize the structure
            structured = self._normalize_structure(structured)
            
            # Add metadata from registry fields
            structured['metadata'] = {
                'healthy_volunteers': inputs.get('healthy_volunteers'),
                'sex': inputs.get('sex'),
                'minimum_age': inputs.get('minimum_age'),
                'maximum_age': inputs.get('maximum_age'),
                'std_ages': inputs.get('std_ages', [])
            }
            
            # Add baseline characteristics if available
            if inputs.get('baseline_characteristics'):
                structured['baseline_characteristics'] = inputs['baseline_characteristics']
            
            # Generate knowledge graph triples from structured data
            structured['kg_triples'] = self._generate_kg_triples(
                trial_data.nct_id, structured
            )
            
            return self._create_success_result({
                'structured_eligibility': structured
            })
            
        except Exception as e:
            logger.error(f"Task 3 failed: {e}")
            return self._create_error_result([str(e)])
    
    def _build_full_prompt(self, eligibility_text: str) -> str:
        """Build the complete two-part prompt."""
        prompt1 = ELIGIBILITY_PROMPT_PART1.format(
            eligibility_text=eligibility_text[:4000]  # Truncate if too long
        )
        return prompt1 + "\n" + ELIGIBILITY_PROMPT_PART2
    
    def _get_empty_structure(self) -> Dict:
        """Return empty structure with proper schema."""
        return {
            "Inclusion": {
                "Medical conditions": [],
                "Pregnancy": "Unknown",
                "Medications or therapy": [],
                "Laboratory values": [],
                "Demographics": {},
                "Other": []
            },
            "Exclusion": {
                "Medical conditions": [],
                "Pregnancy": "Unknown",
                "Medications or therapy": [],
                "Laboratory values": [],
                "Demographics": {},
                "Other": []
            },
            "Summary": "No eligibility criteria available"
        }
    
    def _normalize_structure(self, structured: Dict) -> Dict:
        """Normalize and validate the parsed structure."""
        # Ensure proper structure exists
        for section in ['Inclusion', 'Exclusion']:
            if section not in structured:
                structured[section] = {}
            
            # Ensure all required fields exist
            defaults = {
                "Medical conditions": [],
                "Pregnancy": "Unknown",
                "Medications or therapy": [],
                "Laboratory values": [],
                "Demographics": {},
                "Other": []
            }
            
            for key, default in defaults.items():
                if key not in structured[section]:
                    structured[section][key] = default
                # Normalize "unknown" strings in lists
                if isinstance(structured[section][key], list):
                    structured[section][key] = [
                        item for item in structured[section][key]
                        if item and item.lower() != "unknown"
                    ]
            
            # Validate Pregnancy field
            pregnancy = structured[section].get("Pregnancy", "Unknown")
            if pregnancy not in ["Yes", "No", "Unknown"]:
                structured[section]["Pregnancy"] = "Unknown"
        
        return structured
    
    def _generate_kg_triples(self, nct_id: str, structured: Dict) -> List[Dict]:
        """Generate knowledge graph triples from structured eligibility."""
        triples = []
        
        # First, validate pregnancy logic
        inclusion_pregnancy = structured.get('Inclusion', {}).get('Pregnancy', 'Unknown')
        exclusion_pregnancy = structured.get('Exclusion', {}).get('Pregnancy', 'Unknown')
        
        # Check for logical conflict: both cannot be Yes or both cannot be No
        # If conflict exists, set both to Unknown
        pregnancy_conflict = False
        if inclusion_pregnancy == exclusion_pregnancy and inclusion_pregnancy in ['Yes', 'No']:
            pregnancy_conflict = True
            # Both are Yes or both are No - this is logically impossible
            # Set to Unknown
            if 'Inclusion' in structured:
                structured['Inclusion']['Pregnancy'] = 'Unknown'
            if 'Exclusion' in structured:
                structured['Exclusion']['Pregnancy'] = 'Unknown'
        
        for section in ['Inclusion', 'Exclusion']:
            relation_prefix = "includes" if section == "Inclusion" else "excludes"
            section_data = structured.get(section, {})
            
            # Medical conditions
            for condition in section_data.get("Medical conditions", []):
                triples.append({
                    'head': nct_id,
                    'relation': f'{relation_prefix}_condition',
                    'tail': condition,
                    'head_type': 'Trial',
                    'tail_type': 'Disease'
                })
            
            # Medications or therapy
            for med in section_data.get("Medications or therapy", []):
                triples.append({
                    'head': nct_id,
                    'relation': f'{relation_prefix}_medication',
                    'tail': med,
                    'head_type': 'Trial',
                    'tail_type': 'Drug'
                })
            
            # Pregnancy - skip if Unknown or if there was a conflict
            pregnancy = section_data.get("Pregnancy", "Unknown")
            if pregnancy != "Unknown" and not pregnancy_conflict:
                triples.append({
                    'head': nct_id,
                    'relation': f'{relation_prefix}_pregnancy',
                    'tail': pregnancy,
                    'head_type': 'Trial',
                    'tail_type': 'PregnancyStatus'
                })
        
        return triples
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare eligibility data for structuring.
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with eligibility information
        """
        eligibility = getattr(trial_data, 'eligibility', {})
        baseline = getattr(trial_data, 'baseline_characteristics', {})
        
        return {
            'eligibility_text': eligibility.get('eligibilityCriteria', ''),
            'healthy_volunteers': eligibility.get('healthyVolunteers'),
            'sex': eligibility.get('sex'),
            'minimum_age': eligibility.get('minimumAge'),
            'maximum_age': eligibility.get('maximumAge'),
            'std_ages': eligibility.get('stdAges', []),
            'baseline_characteristics': self._extract_baseline(baseline)
        }
    
    def _extract_baseline(self, baseline: Dict) -> List[Dict]:
        """Extract baseline characteristics."""
        characteristics = []
        
        measures = baseline.get('measures', [])
        for measure in measures:
            char = {
                'title': measure.get('title'),
                'param_type': measure.get('paramType'),
                'unit': measure.get('unitOfMeasure'),
                'categories': []
            }
            
            for category in measure.get('classes', []):
                char['categories'].append({
                    'title': category.get('title'),
                    'values': category.get('categories', [])
                })
            
            characteristics.append(char)
        
        return characteristics
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract structured eligibility."""
        import re
        
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse eligibility JSON: {e}")
        
        return {
            'error': 'Failed to parse response',
            'raw_response': response[:500]
        }


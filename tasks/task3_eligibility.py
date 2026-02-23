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

from typing import Dict, List, Any, Optional
import json
import logging
import re

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
- For `entity_qualifiers`, `core_entity` MUST be pure concept only:
  - Do NOT include thresholds, numbers, units, time windows, grades, or conditional clauses in `core_entity`
  - Put all constraints in `qualifiers.numeric_constraint`, `qualifiers.temporal`, `qualifiers.severity_grade`, `qualifiers.conditional_clause`
  - `core_entity` and qualifier values must not overlap semantically
  - If text is assigned to `qualifiers.conditional_clause`, that same text MUST NOT appear in `core_entity`
  - Remove trailing condition tails from `core_entity`, especially phrases starting with `with`, `without`, `if`, `unless`, `when`, `where`

# Input Eligibility Criteria
{eligibility_text}
"""

ELIGIBILITY_PROMPT_PART2 = """
# Output Format
Generate a valid JSON object in the following structure:

{{
  "entity_qualifiers": [
    {{
      "section": "Inclusion" | "Exclusion",
      "slot": "Medical conditions" | "Medications or therapy" | "Laboratory values" | "Other" | "Pregnancy" | "Demographics",
      "original_text": "exact text span",
      "core_entity": "normalized core entity without qualifiers (no numbers, no operators, no units, no time window, no condition, and no overlap with conditional_clause)",
      "qualifiers": {{
        "polarity": "include" | "exclude" | "required" | "prohibited" | "allowed" | "unknown",
        "entity_type": "disease" | "therapy" | "biomarker" | "vital_sign" | "functional_status" | "procedure" | "other",
        "status": "active" | "prior" | "recent" | "ongoing" | "unstable" | "corrected" | "uncontrolled" | "recurrent" | "unknown",
        "temporal": {{
          "window_value": 0,
          "window_unit": "day|days|week|weeks|month|months|year|years",
          "window_direction": "within_last|since|before|after"
        }} | null,
        "numeric_constraint": {{
          "operator": "<|<=|>|>=|=|range",
          "value": "number or ratio",
          "unit": "unit"
        }} | null,
        "severity_grade": "e.g., ECOG 0-2, NYHA Class II-IV" | null,
        "prior_treatment": {{
          "required": true | false,
          "prohibited": true | false,
          "line_or_regimen": "text or null"
        }} | null,
        "conditional_clause": "if ... clause or null",
        "evidence_or_rule_source": "e.g., RECIST v1.1, CTCAE v5.0, ECOG" | null
      }}
    }}
  }}
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

            # Keep a minimal output schema centered on entity_qualifiers.
            if not isinstance(structured, dict):
                structured = {}
            
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

            # Normalize entity_qualifiers returned by the same LLM response.
            structured['entity_qualifiers'] = self._normalize_entity_qualifiers(
                structured.get('entity_qualifiers')
            )
            
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
        return {"entity_qualifiers": []}
    
    def _normalize_entity_qualifiers(self, entity_qualifiers: Any) -> List[Dict]:
        """Validate/normalize entity_qualifiers shape returned by LLM."""
        if not isinstance(entity_qualifiers, list):
            return []

        normalized: List[Dict] = []
        for item in entity_qualifiers:
            if not isinstance(item, dict):
                continue
            qualifiers = item.get('qualifiers')
            if not isinstance(qualifiers, dict):
                qualifiers = {}

            normalized_item = {
                'section': item.get('section'),
                'slot': item.get('slot'),
                'original_text': item.get('original_text'),
                'core_entity': item.get('core_entity'),
                'qualifiers': {
                    'polarity': qualifiers.get('polarity', 'unknown'),
                    'entity_type': qualifiers.get('entity_type', 'other'),
                    'status': qualifiers.get('status', 'unknown'),
                    'temporal': qualifiers.get('temporal'),
                    'numeric_constraint': qualifiers.get('numeric_constraint'),
                    'severity_grade': qualifiers.get('severity_grade'),
                    'prior_treatment': qualifiers.get('prior_treatment'),
                    'conditional_clause': qualifiers.get('conditional_clause'),
                    'evidence_or_rule_source': qualifiers.get('evidence_or_rule_source'),
                }
            }

            # Enforce non-overlap: strip numeric/time/condition tokens from core_entity.
            normalized_item['core_entity'] = self._strip_core_entity_overlap(
                normalized_item.get('core_entity'),
                normalized_item['qualifiers']
            )
            normalized.append(normalized_item)
        return normalized

    def _strip_core_entity_overlap(self, core_entity: Any, qualifiers: Dict) -> Optional[str]:
        """Remove qualifier-like fragments from core_entity to keep concept/qualifier separation."""
        if not isinstance(core_entity, str):
            return core_entity
        text = core_entity.strip()
        if not text:
            return text

        # Remove numeric/operator/unit patterns (e.g., >= 12 weeks, <140/90 mmHg).
        text = re.sub(r'(<=|>=|<|>|≤|≥|=)\s*\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*[a-zA-Z%/]*', '', text)
        text = re.sub(r'\b\d+(?:\.\d+)?\s*(day|days|week|weeks|month|months|year|years|mmhg|kg|g|mg|ml|l)\b', '', text, flags=re.IGNORECASE)

        # Remove common qualifier markers.
        text = re.sub(r'\bif\b.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(within|since|before|after)\b.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(ECOG|NYHA|RECIST|CTCAE)\b.*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(with|without|unless|when|where)\b.*$', '', text, flags=re.IGNORECASE)

        # Remove exact conditional clause overlap if provided.
        cond = qualifiers.get('conditional_clause') if isinstance(qualifiers, dict) else None
        if isinstance(cond, str) and cond.strip():
            cond_clean = cond.strip().strip(" ,;.-[]()")
            if cond_clean:
                text = re.sub(re.escape(cond_clean), '', text, flags=re.IGNORECASE)

        # Clean residual punctuation/whitespace.
        text = re.sub(r'\s+', ' ', text).strip(" ,;.-[]()")
        return text or core_entity.strip()
    
    def _generate_kg_triples(self, nct_id: str, structured: Dict) -> List[Dict]:
        """Generate knowledge graph triples from structured eligibility."""
        triples = []

        slot_relation_map = {
            "Medical conditions": ("condition", "Disease"),
            "Medications or therapy": ("medication", "Drug"),
            "Laboratory values": ("lab_value", "LabCriterion"),
            "Other": ("criterion", "EligibilityCriterion"),
            "Pregnancy": ("pregnancy", "PregnancyStatus"),
            "Demographics": ("demographic", "DemographicCriterion"),
        }

        for record in structured.get('entity_qualifiers', []) or []:
            if not isinstance(record, dict):
                continue
            section = record.get('section')
            slot = record.get('slot')
            if section not in ("Inclusion", "Exclusion") or slot not in slot_relation_map:
                continue

            relation_prefix = "includes" if section == "Inclusion" else "excludes"
            relation_suffix, tail_type = slot_relation_map[slot]
            tail = record.get('core_entity') or record.get('original_text')
            if not tail or self._is_unknown_tail_value(tail):
                continue

            triples.append({
                'head': nct_id,
                'relation': f'{relation_prefix}_{relation_suffix}',
                'tail': tail,
                'head_type': 'Trial',
                'tail_type': tail_type,
                'tail_attributes': (record.get('qualifiers') if isinstance(record.get('qualifiers'), dict) else {})
            })

        # Baseline characteristics triples
        # Rule:
        # - head: NCTID + "_" + last 3 digits of groupId (e.g., NCT02119676_000)
        # - relation: baseline_characteristics
        # - tail: deepest category title (e.g., "American Indian or Alaska Native")
        # - COUNT-type only included when value > 0
        # - attributes: value, unit, param_type
        baseline = structured.get('baseline_characteristics', [])
        triples.extend(self._generate_baseline_triples(nct_id, baseline))
        
        return triples

    def _is_unknown_tail_value(self, value: Any) -> bool:
        """Return True for placeholder/empty tail values that should not form triples."""
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        v = value.strip().lower()
        return v in {"", "unknown", "n/a", "na", "null", "none", "not applicable"}

    def _generate_baseline_triples(self, nct_id: str, baseline_characteristics: List[Dict]) -> List[Dict]:
        """Generate baseline characteristic triples based on nested baseline structure."""
        triples = []

        for measure in baseline_characteristics or []:
            measure_title = measure.get('title')
            param_type = measure.get('param_type')
            unit = measure.get('unit')

            for cls in measure.get('categories', []) or []:
                for value_block in cls.get('values', []) or []:
                    tail_title = value_block.get('title') or measure_title

                    for m in value_block.get('measurements', []) or []:
                        group_id = m.get('groupId', '')
                        value_raw = m.get('value')

                        if not group_id or value_raw is None or tail_title is None:
                            continue

                        if self._is_count_type(param_type):
                            if not self._is_positive_number(value_raw):
                                continue

                        head = self._build_baseline_head(nct_id, group_id)
                        if not head:
                            continue

                        relation = self._build_baseline_relation(tail_title)

                        triples.append({
                            'head': head,
                            'relation': relation,
                            'tail': str(value_raw),
                            'head_type': 'TrialGroup',
                            'tail_type': 'Value',
                            'tail_attributes': {
                                'unit': unit,
                                'param_type': param_type,
                                'title': tail_title
                            }
                        })

        return triples

    def _build_baseline_head(self, nct_id: str, group_id: str) -> str:
        """Build head entity as NCTID_XXX where XXX is the last 3 digits from group id."""
        import re
        match = re.search(r'(\d{3})$', str(group_id))
        if not match:
            return ""
        return f"{nct_id}_{match.group(1)}"

    def _is_count_type(self, param_type: str) -> bool:
        """Check whether param type indicates count."""
        if not param_type:
            return False
        p = str(param_type).upper()
        return 'COUNT' in p or p == 'NUMBER'

    def _is_positive_number(self, value) -> bool:
        """Return True when numeric value is greater than zero."""
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    def _build_baseline_relation(self, title: str) -> str:
        """Build normalized relation name: baseline_characteristics_<title>."""
        import re
        clean = re.sub(r'[^a-zA-Z0-9]+', '_', str(title).strip()).strip('_')
        return f"baseline_characteristics_{clean}" if clean else "baseline_characteristics"
    
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

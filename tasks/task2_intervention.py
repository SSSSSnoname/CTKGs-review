"""
Task 2: Profiling Complex Interventions

Normalizes free-text intervention descriptions into eight standardized attributes:
- Name
- Type
- Dosage form
- Route
- Dose
- Frequency
- Duration
- Administration sequence

Data source: ResultsSection.outcomeMeasuresModule.outcomeMeasures[].groups[]
These outcome groups contain detailed intervention descriptions including dosage and regimen.

Also models multi-component regimens with:
- co-treat: therapies administered together
- composite: membership within a single intervention arm

The extraction process identifies:
- Drug names (standardized official names)
- Drug types (drug, biological, etc.)
- Dosage forms (tablet, injection, capsule, etc.)
- Administration routes (oral, intravenous, etc.)
- Dosage amounts with standardized units
- Frequency of administration
- Treatment duration (converted to days)
- Administration sequence between multiple drugs
"""

from typing import Dict, List, Any
import json
import logging
import re

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

INTERVENTION_PROMPT_TEMPLATE = """
# Role
You are a Biomedical Drug Information Extraction Engine.

# Task
Extract drug or therapy information from clinical trial group descriptions and output in JSON format.

# Instructions
1. Extract the "name", "type", "dosage_form", "administration_route", "dosage", "frequency", "Treatment duration", and an "id" for each drug or therapy.
2. If there is an administration sequence, include a separate key called "administration_sequence" that outputs the drug ids in the correct order. If no sequence is provided, return "administration_sequence": "no order".
3. Use "+" to indicate combination drugs, ">>" to indicate the order of administration, "/" to indicate choices between drugs, and "()" to indicate drugs used together.
4. Do not use abbreviations. If a drug/therapy is a combination, output every drug/therapy separately.
5. Carefully distinguish placebo treatments. Expressions such as "placebo", "placebo of X", or "X placebo" all refer to a placebo. In such cases, the name should be "placebo", and the specific reference (e.g., placebo for drug X) can be described in the dosage_form field, for example: "tablet (placebo for X)".

# Format Requirements
- The "name" field must use the standard official name of the drug or therapy. Do NOT use abbreviations, do NOT add supplementary information, and do NOT modify the name. Use only the standard official name as it appears in drug databases (e.g., "Metformin" not "Met", "Aspirin" not "ASA"). If only a drug category or class is mentioned (e.g., "atypical antipsychotic", "oral atypical antipsychotic", "beta-blocker"), use that category name as the "name" field instead of "NA".
- The "dosage" field must include a standardized unit. Acceptable units include: mg (milligrams), g (grams), ml (milliliters), L (liters), IU (International Units), mcg or μg (micrograms), units, % (percentage for certain formulations). Always convert to the most appropriate standard unit (e.g., convert grams to mg if less than 1g, convert liters to ml if less than 1L).
- The "Treatment duration" field must ALWAYS be converted to days as the unit. Convert weeks to days (1 week = 7 days), months to days (1 month = 30 days), years to days (1 year = 365 days). Format as "X days" (e.g., "7 days", "14 days", "30 days").
- The "frequency" field can be expressed in either hours or days as the unit. Use hours for frequencies less than 24 hours (e.g., "every 8 hours", "every 12 hours"), and use days for daily or less frequent administration (e.g., "once a day", "twice a day", "every 2 days", "once a week").
- If a field is missing, return "NA" for that field.
- The final JSON should contain a separate dictionary for the administration sequence.
- The "original_text" field must be the original text from the provided text.

# Output Format
Provide a valid JSON object:
{{
  "Interventions": [
    {{
      "name": "Standard official drug name",
      "original_text": "Original name from the provided text",
      "type": "drug",
      "dosage_form": "tablet/injection/capsule/etc or NA",
      "administration_route": "oral/intramuscular/intravenous/etc or NA",
      "dosage": "X mg/ml/g/etc",
      "frequency": "every X hours/days or once/twice a day",
      "Treatment duration": "X days",
      "id": "01"
    }}
  ],
  "administration_sequence": "01>>02" or "no order"
}}

EXAMPLE_GROUP_INFO:
title: 'L+M 1000 Fed', description: '5 mg linagliptin 7days then 1000 mg metformin XR (given as 1 tablet 5 mg linagliptin and 2 tablets 500 mg metformin XR) oral with 240 mL of water after a high-fat, high-calorie meal 24 days.'


EXAMPLE_OUTPUT:
{
    "Interventions": [
        {
            "name": "Linagliptin",
            "original_text": "linagliptin",
            "type": "drug",
            "dosage_form": "tablet",
            "administration_route": "oral",
            "dosage": "5 mg",
            "frequency": "once a day",
            "Treatment duration": "7 days",
            "id": "01"
        },
        {
            "name": "Metformin extended-release",
            "original_text": "metformin XR",
            "type": "drug",
            "dosage_form": "tablet",
            "administration_route": "oral",
            "dosage": "1000 mg",
            "frequency": "once a day",
            "Treatment duration": "24 days",
            "id": "02"
        }
    ],
    "administration_sequence": "01>>02"
}

# Input Data

Group's information:
Title: {title}
Description: {description}
"""


# =============================================================================
# PROMPT AND PARSING FUNCTIONS
# =============================================================================

def get_intervention_prompt(group_data: Dict) -> str:
    """
    Generate the complete intervention extraction prompt.
    
    Args:
        group_data: Dictionary containing:
            - title: Group title
            - description: Group description
            - trialsummary: Trial summary (optional)
    
    Returns:
        Complete prompt string ready for LLM input
    """
    trialsummary = str(group_data.get('trialsummary', '') or '')
    title = str(group_data.get('title', '') or '')
    description = str(group_data.get('description', '') or '')
    
    # Use string replacement to avoid brace conflicts in the template
    prompt = INTERVENTION_PROMPT_TEMPLATE.replace('{trialsummary}', '___TRIALSUMMARY___')
    prompt = prompt.replace('{title}', '___TITLE___')
    prompt = prompt.replace('{description}', '___DESCRIPTION___')
    
    prompt = prompt.replace('___TRIALSUMMARY___', trialsummary)
    prompt = prompt.replace('___TITLE___', title)
    prompt = prompt.replace('___DESCRIPTION___', description)
    
    return prompt


def parse_intervention_response(response_content: str) -> Dict:
    """
    Parse LLM response to extract intervention information.
    
    Args:
        response_content: Raw response content from LLM
    
    Returns:
        Parsed dictionary containing interventions and administration sequence,
        or error information if parsing fails
    """
    json_match = re.search(r'\{[\s\S]*\}', response_content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_content": response_content[:500]
            }
    return {
        "error": "No JSON found in response",
        "raw_content": response_content[:500]
    }


class Task2InterventionProfiling(BaseTaskHandler):
    """
    Task 2: Profiling Complex Interventions
    
    Extracts intervention data from ResultsSection outcome groups,
    which contain detailed descriptions including dosage and regimen.
    """
    
    @property
    def task_name(self) -> str:
        return "task2_intervention_profiling"
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute intervention profiling for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with profiled interventions
        """
        try:
            # Prepare input data from ResultsSection
            inputs = self.prepare_input(trial_data)
            
            outcome_groups = inputs.get('outcome_groups', [])
            
            if not outcome_groups:
                # Fallback to protocol section if no results
                logger.info("No outcome groups found in ResultsSection, falling back to protocol section")
                return self._execute_from_protocol(trial_data, inputs)
            
            # Process each outcome group
            profiled_interventions = []
            errors = []
            
            for group_input in outcome_groups:
                try:
                    # Generate prompt using existing intervention extraction
                    prompt = get_intervention_prompt(group_input)
                    
                    # Call LLM
                    response = self.call_llm(prompt)
                    
                    # Parse response
                    parsed = parse_intervention_response(response)
                    
                    normalized_group_id = self._normalize_group_id(
                        group_input.get('group_id'),
                        nct_id=trial_data.nct_id
                    )
                    profiled_interventions.append({
                        'group_id': normalized_group_id,
                        'original_group_id': group_input.get('group_id'),
                        'group_title': group_input.get('title'),
                        'original_description': group_input.get('description'),
                        'interventions': parsed.get('Interventions', []),
                        'administration_sequence': parsed.get('administration_sequence', 'no order')
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process group {group_input.get('group_id')}: {e}")
                    errors.append(f"{group_input.get('group_id')}: {str(e)}")
            
            # Build co-treat and composite relations
            relations = self._build_relations(profiled_interventions, trial_data.nct_id)
            
            if errors and not profiled_interventions:
                return self._create_error_result(errors)
            
            return self._create_success_result({
                'profiled_interventions': profiled_interventions,
                'co_treat_relations': relations.get('co_treat', []),
                'composite_relations': relations.get('composite', []),
                'total_groups': len(outcome_groups),
                'processed_groups': len(profiled_interventions),
                'data_source': 'ResultsSection.outcomeMeasuresModule',
                'errors': errors
            })
            
        except Exception as e:
            logger.error(f"Task 2 failed: {e}")
            return self._create_error_result([str(e)])
    
    def _execute_from_protocol(self, trial_data: Any, inputs: Dict) -> TaskResult:
        """
        Fallback: Execute using protocol section arm groups.
        Used when trial has no results section.
        """
        arm_groups = inputs.get('protocol_arm_groups', [])
        
        if not arm_groups:
            return self._create_success_result({
                'profiled_interventions': [],
                'message': 'No intervention groups found in protocol or results'
            })
        
        profiled_interventions = []
        errors = []
        
        for arm_input in arm_groups:
            try:
                prompt = get_intervention_prompt(arm_input)
                response = self.call_llm(prompt)
                parsed = parse_intervention_response(response)
                
                normalized_group_id = self._normalize_group_id(
                    arm_input.get('group_id'),
                    nct_id=trial_data.nct_id
                )
                profiled_interventions.append({
                    'group_id': normalized_group_id,
                    'original_group_id': arm_input.get('group_id'),
                    'group_title': arm_input.get('title'),
                    'original_description': arm_input.get('description'),
                    'interventions': parsed.get('Interventions', []),
                    'administration_sequence': parsed.get('administration_sequence', 'no order')
                })
            except Exception as e:
                logger.error(f"Failed to process arm {arm_input.get('group_id')}: {e}")
                errors.append(str(e))
        
        relations = self._build_relations(profiled_interventions, trial_data.nct_id)
        
        return self._create_success_result({
            'profiled_interventions': profiled_interventions,
            'co_treat_relations': relations.get('co_treat', []),
            'composite_relations': relations.get('composite', []),
            'total_groups': len(arm_groups),
            'processed_groups': len(profiled_interventions),
            'data_source': 'protocolSection.armsInterventionsModule',
            'errors': errors
        })
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare intervention data for profiling.
        
        Primary source: ResultsSection.outcomeMeasuresModule.outcomeMeasures[].groups[]
        Fallback: protocolSection.armsInterventionsModule.armGroups[]
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with 'outcome_groups', 'protocol_arm_groups', and 'trial_summary'
        """
        description_module = getattr(trial_data, 'description', {})
        trial_summary = description_module.get('briefSummary', '')
        
        # Primary: Extract from ResultsSection
        outcome_groups = self._extract_outcome_groups(trial_data, trial_summary)
        
        # Fallback: Extract from protocol section
        protocol_arm_groups = self._extract_protocol_arms(trial_data, trial_summary)
        
        return {
            'outcome_groups': outcome_groups,
            'protocol_arm_groups': protocol_arm_groups,
            'trial_summary': trial_summary
        }
    
    def _extract_outcome_groups(self, trial_data: Any, trial_summary: str) -> List[Dict]:
        """
        Extract outcome groups from ResultsSection.
        
        These groups contain detailed intervention descriptions including:
        - Drug names
        - Dosages (e.g., "15 mg BID")
        - Frequencies (e.g., "continuous", "for first 21 days")
        - Cycle information (e.g., "28-day cycle")
        """
        outcome_groups = []
        seen_groups = set()  # Avoid duplicates across outcome measures
        
        # Use outcome_measures attribute from TrialData
        outcome_measures_module = getattr(trial_data, 'outcome_measures', {})
        if not outcome_measures_module:
            return []
        
        outcome_measures = outcome_measures_module.get('outcomeMeasures', [])
        
        for om in outcome_measures:
            groups = om.get('groups', [])
            for group in groups:
                group_id = group.get('id', '')
                
                # Skip if already processed
                if group_id in seen_groups:
                    continue
                seen_groups.add(group_id)
                
                title = group.get('title', '')
                description = group.get('description', '')
                
                if not description:
                    continue
                
                outcome_groups.append({
                    'group_id': group_id,
                    'title': title,
                    'description': description,
                    'trialsummary': trial_summary
                })
        
        return outcome_groups
    
    def _extract_protocol_arms(self, trial_data: Any, trial_summary: str) -> List[Dict]:
        """
        Extract arm groups from protocol section as fallback.
        """
        arm_groups = []
        
        arms_module = getattr(trial_data, 'arms_interventions', {})
        
        # Get intervention descriptions to enrich arm data
        interventions = arms_module.get('interventions', [])
        intervention_by_name = {}
        for intv in interventions:
            name = intv.get('name', '').lower()
            intervention_by_name[name] = intv.get('description', '')
        
        for idx, arm in enumerate(arms_module.get('armGroups', [])):
            arm_desc = arm.get('description', '')
            
            # If arm has no description, build from intervention descriptions
            if not arm_desc:
                intervention_names = arm.get('interventionNames', [])
                desc_parts = []
                for int_name in intervention_names:
                    # Handle "Type: Name" format
                    clean_name = int_name.lower()
                    if ': ' in clean_name:
                        clean_name = clean_name.split(': ', 1)[1]
                    if clean_name in intervention_by_name:
                        desc_parts.append(f"{int_name}: {intervention_by_name[clean_name]}")
                arm_desc = '\n'.join(desc_parts)
            
            arm_groups.append({
                'group_id': f"AG{idx:03d}",
                'title': arm.get('label', ''),
                'description': arm_desc,
                'type': arm.get('type', ''),
                'trialsummary': trial_summary
            })
        
        return arm_groups
    
    def _normalize_group_id(self, group_id: str, nct_id: str = "") -> str:
        """
        Normalize group ID to unified format.
        
        BG000, OG000, EG000, AG000 -> NCTID_000 (if nct_id is provided)
        BG000, OG000, EG000, AG000 -> Group_000 (fallback)
        
        Args:
            group_id: Original group ID
            
        Returns:
            Normalized group ID in format Group_XXX
        """
        if not group_id:
            return group_id
        
        import re
        # Extract the numeric part
        # Handles: BG000, OG000, EG000, AG000, etc.
        match = re.match(r'[A-Z]{1,2}(\d+)', group_id)
        if match:
            num = match.group(1)
            if nct_id:
                return f"{nct_id}_{num}"
            return f"Group_{num}"
        
        return group_id
    
    def _build_relations(self, profiled_interventions: List[Dict], nct_id: str = "") -> Dict[str, List]:
        """
        Build co-treat and composite relations from profiled interventions.
        
        co_treat: Pairwise relations between drugs administered together
        composite: Relation between drug and its containing group
        
        Returns:
            Dictionary with 'co_treat' and 'composite' relation lists
        """
        co_treat = []
        composite = []
        
        for arm in profiled_interventions:
            interventions = arm.get('interventions', [])
            group_id = arm.get('group_id', '')
            # Normalize group ID: OG000 -> Group_000
            normalized_group = self._normalize_group_id(group_id, nct_id=nct_id)
            group_title = arm.get('group_title', '')
            sequence = arm.get('administration_sequence', 'no order')
            
            if len(interventions) == 0:
                continue

            def _entity_attrs(intervention_obj: Dict) -> Dict:
                raw_intervention_id = str(intervention_obj.get('id', '') or '').strip()
                normalized_intervention_id = (
                    f"{normalized_group}_{raw_intervention_id}"
                    if raw_intervention_id else None
                )
                return {
                    'type': intervention_obj.get('type'),
                    'dosage_form': intervention_obj.get('dosage_form'),
                    'administration_route': intervention_obj.get('administration_route'),
                    'dosage': intervention_obj.get('dosage'),
                    'frequency': intervention_obj.get('frequency'),
                    'Treatment duration': intervention_obj.get('Treatment duration'),
                    'id': normalized_intervention_id
                }
            
            # Build composite relations: each drug belongs to this group
            for intervention in interventions:
                composite.append({
                    'head': intervention.get('name'),
                    'relation': 'belongs_to_group',
                    'tail': normalized_group,
                    'head_type': 'Drug',
                    'tail_type': 'Group',
                    'head_attributes': _entity_attrs(intervention),
                    'tail_attributes': {
                        'group_id': normalized_group,
                        'group_title': group_title,
                        'administration_sequence': sequence
                    }
                })
            
            # Build co-treat relations: pairwise drug combinations
            if len(interventions) >= 2:
                if sequence == 'no order':
                    # All interventions are co-administered - create pairwise relations
                    for i, int1 in enumerate(interventions):
                        for int2 in interventions[i+1:]:
                            co_treat.append({
                                'head': int1.get('name'),
                                'relation': 'co_treat',
                                'tail': int2.get('name'),
                                'head_type': 'Drug',
                                'tail_type': 'Drug',
                                'head_attributes': _entity_attrs(int1),
                                'tail_attributes': _entity_attrs(int2),
                                'relation_attributes': {
                                    'group_id': normalized_group,
                                    'group_title': group_title,
                                    'administration_sequence': sequence
                                }
                            })
                else:
                    # Sequential administration - still create co-treat with sequence info
                    for i, int1 in enumerate(interventions):
                        for int2 in interventions[i+1:]:
                            co_treat.append({
                                'head': int1.get('name'),
                                'relation': 'co_treat',
                                'tail': int2.get('name'),
                                'head_type': 'Drug',
                                'tail_type': 'Drug',
                                'head_attributes': _entity_attrs(int1),
                                'tail_attributes': _entity_attrs(int2),
                                'relation_attributes': {
                                    'group_id': normalized_group,
                                    'group_title': group_title,
                                    'administration_sequence': sequence
                                }
                            })
        
        return {
            'co_treat': co_treat,
            'composite': composite
        }

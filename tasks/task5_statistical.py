"""
Task 5: Extracting Intervention-Outcome Conclusions

Converts posted statistical analyses into conclusion-level triples:
- Non-comparative analyses (single-arm significance)
- Comparative analyses (relative efficacy across arms)

Uses multi-strategy prompting:
- Open prompts
- Stepwise reasoning
- Self-consistency

Produces effect-direction labels and significance judgments.

This module provides prompt templates and processing functions for a two-step
statistical analysis of clinical trial outcome data using LLM.

Step 1: Generate comparative triplets
- Compare two groups and determine which has greater/lesser degree of change
- Handle different analysis types: TWO_GROUP, SINGLE_GROUP, NON_INFERIORITY

Step 2: Validate statistical significance
- Verify the statistical significance of the comparison from Step 1
- Analyze p-values, confidence intervals, and other statistical measures
"""

from typing import Dict, List, Any
import json
import logging
import re

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 PROMPT TEMPLATES
# =============================================================================

PROMPT_TWO_GROUP = """
# Role
You are a Biomedical Logic Engine.

# Task
Compare two clinical trial groups and determine which group has a greater degree of change (increase or decrease) for the given outcome.

# Instructions
1. Read the outcome description to understand what is being measured and whether higher or lower values are better.
2. Compare the numerical values of the two groups.
3. Determine which group shows a greater or lower degree of the desired change (increase for positive outcomes, decrease for negative outcomes).
4. Express the relationship: which group has greater/lower degree than the other in increasing/decreasing/improving the outcome.
5. **IMPORTANT**: Verify that the head_entity_id, head_entity_2_id, head_entity_name, and head_entity_2_name you use in the triplet are exactly the same as the Group IDs and Group Names that appear in the Input Data. Do not create or invent group IDs or names that are not present in the Input Data.

# Output Format
Provide a valid JSON object using Group IDs as entities:
{   "logic_trace": {
    "outcome_polarity": "NEGATIVE or POSITIVE",
    "head_entity_value": number,
    "head_entity_2_value": number,
    "interpretation": "Why chosse this relationship_context and comparative_relation"
  },
  "triplet": {
    "head_entity_id": "Group ID with greater degree (e.g., OG000)",
    "head_entity_name": "Original Name of Group 1",
    "head_entity_2_id": "Group ID with lesser degree (e.g., OG001)",
    "head_entity_2_name": "Original Name of Group 2",
    "comparative_relation": "has greater/lower degree than",
    "relationship_context": "in decreasing" | "in increasing" | "in improving" (consider the outcome polarity),
    "tail_entity_original": "Outcome Original Name"
  }
}

# Input Data
{Input data}
"""

PROMPT_NON_INFERIORITY = """
# Role
You are a Biomedical Logic Engine for Non-Inferiority Trials.

# Task
In non-inferiority trials, determine if the new intervention shows equivalent or better performance compared to the reference/control group for the given outcome.

# Instructions
1. Identify which group is the new intervention and which is the reference/control.
2. Read the outcome description to understand what is being measured.
3. Compare the numerical values of the two groups.
4. Determine if the new intervention is equivalent to or better than the reference in terms of the outcome.
5. **IMPORTANT**: Verify that the head_entity_id, head_entity_2_id, head_entity_name, and head_entity_2_name you use in the triplet are exactly the same as the Group IDs and Group Names that appear in the Input Data. Do not create or invent group IDs or names that are not present in the Input Data.

# Output Format
Provide a valid JSON object using Group IDs as entities:
{   "logic_trace": {
    "outcome_polarity": "NEGATIVE or POSITIVE",
    "head_entity_value": number,
    "head_entity_2_value": number,
    "interpretation": "Why chosse this relationship_context and comparative_relation"
  },
  "triplet": {
    "head_entity_id": "Group ID of New Intervention (e.g., OG000)",
    "head_entity_name": "Original Name of New Intervention",
    "head_entity_2_id": "Group ID of Reference/Control (e.g., OG001)",
    "head_entity_2_name": "Original Name of Reference/Control",
    "comparative_relation": "has equivalent / greater / lower degree than",
    "relationship_context": "in decreasing" | "in increasing" | "in improving",
    "tail_entity_original": "Outcome Original Name"
  }
}

# Input Data
{Input data}
"""

PROMPT_SINGLE_GROUP = """
# Role
You are a Biomedical Descriptive Analysis Engine.

# Task
Analyze a single group outcome measure. Since there is no comparison, focus on describing the outcome value and its meaning.

# Instructions
1. Read the outcome description to understand what is being measured and whether higher or lower values are better.
2. Extract the outcome value for the single group.
3. Determine if the value represents a good or poor outcome based on the outcome polarity.
4. **IMPORTANT**: Verify that the head_entity_id and head_entity_name you use in the triplet are exactly the same as the Group ID and Group Name that appear in the Input Data. Do not create or invent group IDs or names that are not present in the Input Data.

# Output Format
Provide a valid JSON object (note: single group analysis has no entity_2):
{   "logic_trace": {
    "outcome_polarity": "NEGATIVE or POSITIVE",
    "group_value": number,
    "interpretation": "Why chosse this relationship_context and comparative_relation"
  },
  "triplet": {
    "head_entity_id": "Group ID (e.g., OG000)",
    "head_entity_name": "Original Name of Group",
    "relationship": "e.g., decreasing, increasing,...",
    "tail_entity_original": "Outcome Original Name"
  }
}

# Input Data
{Input data}
"""


# =============================================================================
# STEP 2 PROMPT TEMPLATE
# =============================================================================

PROMPT_STEP2 = """
# Role
You are a Bio-Statistician Validator.

# Task
Validate the statistical significance of the comparison defined in the Reference Triplet.

Based on the statistical information, determine whether this triplet is statistically significant and provide a brief explanation.

# Output Format
{
    "statistical_conclusion": "Brief explanation (e.g., 'Numerically better, but p=0.55 indicates no significant difference' or 'Non-inferiority demonstrated with CI within margin')"
    "is_statistically_significant": boolean
}

# Reference Triplet
{step_1_triplet}

# Statistical Analysis Information
{step_2_input}
"""


# =============================================================================
# PROMPT SELECTION AND FORMATTING FUNCTIONS
# =============================================================================

def get_prompt_for_analysis_type(analysis_type: str, is_non_inferiority: bool = False) -> str:
    """
    Get the appropriate Step 1 prompt based on analysis type.
    
    Args:
        analysis_type: Type of analysis (SINGLE_GROUP, TWO_GROUP)
        is_non_inferiority: Whether this is a non-inferiority trial
    
    Returns:
        Prompt template string
    
    Raises:
        ValueError: If analysis_type is not supported
    """
    if analysis_type == "SINGLE_GROUP":
        return PROMPT_SINGLE_GROUP
    elif analysis_type == "TWO_GROUP":
        if is_non_inferiority:
            return PROMPT_NON_INFERIORITY
        else:
            return PROMPT_TWO_GROUP
    else:
        raise ValueError(
            f"Unknown analysis type: {analysis_type}. "
            "Only SINGLE_GROUP and TWO_GROUP are supported."
        )


def format_step1_data(step1_data: Dict) -> str:
    """
    Format Step 1 input data as text for the LLM prompt.
    
    Args:
        step1_data: Dictionary containing:
            - trial_summary: Brief summary of the trial (optional)
            - outcome: Outcome information (title, description, type, etc.)
            - groups: List of group information with values
    
    Returns:
        Formatted text string for prompt input
    """
    lines = []
    
    # Trial Summary
    trial_summary = step1_data.get('trial_summary', '')
    if trial_summary:
        lines.append("## Trial Summary")
        lines.append(trial_summary)
        lines.append("")
    
    # Outcome Information
    outcome = step1_data.get('outcome', {})
    lines.append("## Outcome Information")
    lines.append(f"Title: {outcome.get('title', 'N/A')}")
    lines.append(f"Description: {outcome.get('description', 'N/A')}")
    lines.append(f"Type: {outcome.get('type', 'N/A')}")
    lines.append(f"Unit of Measure: {outcome.get('unit_of_measure', 'N/A')}")
    lines.append(f"Time Frame: {outcome.get('time_frame', 'N/A')}")
    if outcome.get('param_type'):
        lines.append(f"Parameter Type: {outcome.get('param_type')}")
    lines.append("")
    
    # Groups Information
    lines.append("## Groups Information")
    groups = step1_data.get('groups', [])
    for group in groups:
        lines.append(f"\n### Group {group.get('group_id', 'N/A')}: {group.get('group_title', 'N/A')}")
        lines.append(f"Intervention Description: {group.get('intervention_description', 'N/A')}")
        
        # Participant count
        participant_count = group.get('participant_count', {})
        if participant_count:
            lines.append(
                f"Participant Count: {participant_count.get('value', 'N/A')} "
                f"{participant_count.get('units', '')}"
            )
        
        # Values
        values = group.get('values', [])
        if values:
            lines.append("Values:")
            for value_info in values:
                value_str = f"  - Value: {value_info.get('value', 'N/A')}"
                if value_info.get('category_title'):
                    value_str += f" (Category: {value_info.get('category_title')})"
                if value_info.get('spread'):
                    value_str += f" [Spread: {value_info.get('spread')}"
                    if value_info.get('spreadType'):
                        value_str += f" ({value_info.get('spreadType')})"
                    value_str += "]"
                lines.append(value_str)
    
    return "\n".join(lines)


def format_step2_data(step2_data: Dict) -> str:
    """
    Format Step 2 input data (statistical analysis) as text for the LLM prompt.
    
    Args:
        step2_data: Dictionary containing statistical_analysis information
    
    Returns:
        Formatted text string for prompt input
    """
    lines = []
    stat_analysis = step2_data.get('statistical_analysis', {})
    
    lines.append("## Statistical Analysis Information")
    
    if stat_analysis.get('statisticalMethod'):
        lines.append(f"Statistical Method: {stat_analysis.get('statisticalMethod')}")
    if stat_analysis.get('pValue'):
        lines.append(f"P-value: {stat_analysis.get('pValue')}")
    if stat_analysis.get('pValueComment'):
        lines.append(f"P-value Comment: {stat_analysis.get('pValueComment')}")
    
    if stat_analysis.get('ciLowerLimit') or stat_analysis.get('ciUpperLimit'):
        ci_lower = stat_analysis.get('ciLowerLimit', 'N/A')
        ci_upper = stat_analysis.get('ciUpperLimit', 'N/A')
        ci_percent = stat_analysis.get('ciPercentileNtiles', '95')
        lines.append(f"Confidence Interval ({ci_percent}%): [{ci_lower}, {ci_upper}]")
    
    if stat_analysis.get('testedNonInferiority') is not False:
        lines.append(f"Non-inferiority Test: {stat_analysis.get('testedNonInferiority')}")
        if stat_analysis.get('nonInferiorityType'):
            lines.append(f"Non-inferiority Type: {stat_analysis.get('nonInferiorityType')}")
    
    if stat_analysis.get('method'):
        lines.append(f"Method: {stat_analysis.get('method')}")
    
    return "\n".join(lines)


def format_groups_values(step1_data: Dict) -> str:
    """
    Format group values for Step 2 input (without participant counts).
    
    Args:
        step1_data: Dictionary containing groups information
    
    Returns:
        Formatted text string with group values
    """
    lines = []
    groups = step1_data.get('groups', [])
    
    if groups:
        lines.append("## Groups Values")
        for group in groups:
            group_id = group.get('group_id', 'N/A')
            group_title = group.get('group_title', 'N/A')
            lines.append(f"\n### Group {group_id}: {group_title}")
            
            values = group.get('values', [])
            if values:
                lines.append("Values:")
                for value_info in values:
                    value_str = f"  - Value: {value_info.get('value', 'N/A')}"
                    if value_info.get('category_title'):
                        value_str += f" (Category: {value_info.get('category_title')})"
                    if value_info.get('spread'):
                        value_str += f" [Spread: {value_info.get('spread')}"
                        if value_info.get('spreadType'):
                            value_str += f" ({value_info.get('spreadType')})"
                        value_str += "]"
                    lines.append(value_str)
    
    return "\n".join(lines)


def get_step1_prompt(
    step1_data: Dict,
    analysis_type: str,
    is_non_inferiority: bool = False
) -> str:
    """
    Generate complete Step 1 prompt with input data.
    
    Args:
        step1_data: Input data for Step 1
        analysis_type: Type of analysis (SINGLE_GROUP, TWO_GROUP)
        is_non_inferiority: Whether this is a non-inferiority trial
    
    Returns:
        Complete prompt string ready for LLM input
    """
    template = get_prompt_for_analysis_type(analysis_type, is_non_inferiority)
    input_text = format_step1_data(step1_data)
    return template.replace("{Input data}", input_text)


def get_step2_prompt(
    step1_triplet: Dict,
    step2_data: Dict,
    step1_data: Dict
) -> str:
    """
    Generate complete Step 2 prompt with triplet and statistical data.
    
    Args:
        step1_triplet: Triplet result from Step 1
        step2_data: Statistical analysis data
        step1_data: Original Step 1 data (for group values)
    
    Returns:
        Complete prompt string ready for LLM input
    """
    triplet_str = json.dumps(step1_triplet, indent=2, ensure_ascii=False)
    
    # Format statistical information
    stat_text = format_step2_data(step2_data)
    # Remove duplicate header
    stat_text = stat_text.replace("## Statistical Analysis Information\n", "", 1).strip()
    
    # Add group values
    groups_text = format_groups_values(step1_data)
    
    if groups_text:
        step2_input_full = stat_text + "\n\n" + groups_text
    else:
        step2_input_full = stat_text
    
    prompt = PROMPT_STEP2.replace("{step_1_triplet}", triplet_str)
    prompt = prompt.replace("{step_2_input}", step2_input_full)
    
    return prompt


# =============================================================================
# RESULT PARSING FUNCTIONS
# =============================================================================

def parse_step1_response(response_content: str) -> Dict:
    """
    Parse Step 1 LLM response to extract triplet.
    
    Args:
        response_content: Raw response content from LLM
    
    Returns:
        Parsed dictionary containing triplet and logic_trace
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


def parse_step2_response(response_content: str) -> Dict:
    """
    Parse Step 2 LLM response to extract statistical significance.
    
    Args:
        response_content: Raw response content from LLM
    
    Returns:
        Parsed dictionary containing is_statistically_significant and conclusion
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


class Task5StatisticalConclusions(BaseTaskHandler):
    """
    Task 5: Extracting Intervention-Outcome Conclusions
    
    Systematically converts posted statistical analyses into
    conclusion-level triples with effect direction and significance.
    """
    
    @property
    def task_name(self) -> str:
        return "task5_statistical_conclusions"
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute statistical conclusion extraction for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters (e.g., previous task results)
            
        Returns:
            TaskResult with extracted conclusions
        """
        try:
            # Prepare input data
            inputs = self.prepare_input(trial_data)
            
            if not inputs.get('outcome_analyses'):
                return self._create_success_result({
                    'conclusions': [],
                    'message': 'No statistical analyses found'
                })
            
            # Process each analysis
            conclusions = []
            errors = []
            
            for analysis in inputs['outcome_analyses']:
                try:
                    conclusion = self._process_analysis(analysis)
                    if conclusion:
                        conclusions.append(conclusion)
                except Exception as e:
                    logger.error(f"Failed to process analysis: {e}")
                    errors.append(str(e))
            
            if errors and not conclusions:
                return self._create_error_result(errors)
            
            # Separate comparative and non-comparative
            comparative = [c for c in conclusions if c.get('analysis_type') == 'comparative']
            non_comparative = [c for c in conclusions if c.get('analysis_type') != 'comparative']
            
            # Generate knowledge graph triples
            nct_id = trial_data.nct_id
            kg_triples = self._generate_kg_triples(nct_id, conclusions)
            
            return self._create_success_result({
                'conclusions': conclusions,
                'comparative_conclusions': comparative,
                'non_comparative_conclusions': non_comparative,
                'kg_triples': kg_triples,
                'total_analyses': len(inputs['outcome_analyses']),
                'processed_analyses': len(conclusions),
                'errors': errors
            })
            
        except Exception as e:
            logger.error(f"Task 5 failed: {e}")
            return self._create_error_result([str(e)])
    
    def _generate_kg_triples(self, nct_id: str, conclusions: List[Dict]) -> List[Dict]:
        """
        Generate knowledge graph triples from statistical conclusions.
        
        Creates triples like:
        - (Intervention, improves/worsens/noEffect, Outcome)
        - (Intervention, comparedTo, Comparator)
        
        Args:
            nct_id: NCT ID
            conclusions: List of conclusion dictionaries
            
        Returns:
            List of triple dictionaries
        """
        triples = []
        
        for conclusion in conclusions:
            triplet = conclusion.get('triplet', {})
            is_significant = conclusion.get('is_statistically_significant', False)
            p_value = conclusion.get('p_value')
            
            intervention = triplet.get('head_entity_name', '')
            comparator = triplet.get('head_entity_2_name', '')
            outcome = triplet.get('tail_entity_original', '')
            relation = triplet.get('comparative_relation', '')
            context = triplet.get('relationship_context', '')
            
            if not intervention or not outcome:
                continue
            
            # Determine relation type based on context and significance
            if is_significant:
                if 'better' in context.lower() or 'improve' in context.lower():
                    relation_type = 'improves'
                elif 'worse' in context.lower() or 'decline' in context.lower():
                    relation_type = 'worsens'
                elif 'increase' in context.lower():
                    relation_type = 'increases'
                elif 'decrease' in context.lower() or 'reduce' in context.lower():
                    relation_type = 'decreases'
                else:
                    relation_type = 'affectsOutcome'
            else:
                relation_type = 'noSignificantEffect'
            
            # Main intervention-outcome triple
            triples.append({
                'head': intervention,
                'relation': relation_type,
                'tail': outcome,
                'head_type': 'Intervention',
                'tail_type': 'Outcome',
                'attributes': {
                    'is_significant': is_significant,
                    'p_value': p_value,
                    'comparator': comparator if comparator else None,
                    'context': context[:200] if context else None
                },
                'provenance': {'nct_id': nct_id}
            })
            
            # Comparison triple if there's a comparator
            if comparator and comparator != intervention:
                triples.append({
                    'head': intervention,
                    'relation': 'comparedTo',
                    'tail': comparator,
                    'head_type': 'Intervention',
                    'tail_type': 'Intervention',
                    'attributes': {
                        'outcome': outcome,
                        'result': 'significant' if is_significant else 'not_significant'
                    },
                    'provenance': {'nct_id': nct_id}
                })
        
        return triples
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare statistical analysis data for processing.
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with outcome analyses
        """
        outcome_analyses = []
        
        results_section = getattr(trial_data, 'results_section', {})
        outcome_measures = results_section.get('outcomeMeasuresModule', {})
        measures = outcome_measures.get('outcomeMeasures', [])
        description = getattr(trial_data, 'description', {})
        
        for measure in measures:
            # Extract groups
            groups = []
            for group in measure.get('groups', []):
                groups.append({
                    'group_id': group.get('id'),
                    'group_title': group.get('title'),
                    'intervention_description': group.get('description', '')
                })
            
            # Extract values for each group
            group_values = {}
            for cls in measure.get('classes', []):
                for cat in cls.get('categories', []):
                    for measurement in cat.get('measurements', []):
                        gid = measurement.get('groupId')
                        if gid:
                            group_values[gid] = {
                                'value': measurement.get('value'),
                                'spread': measurement.get('spread'),
                                'spread_type': measurement.get('spreadType'),
                                'category_title': cat.get('title')
                            }
            
            # Add values to groups
            for group in groups:
                gid = group['group_id']
                if gid in group_values:
                    group['values'] = [group_values[gid]]
            
            # Extract statistical analyses
            analyses = measure.get('analyses', [])
            
            for analysis in analyses:
                outcome_analysis = {
                    'outcome_title': measure.get('title'),
                    'outcome_description': measure.get('description'),
                    'outcome_type': measure.get('type'),
                    'unit_of_measure': measure.get('unitOfMeasure'),
                    'param_type': measure.get('paramType'),
                    'time_frame': measure.get('timeFrame'),
                    'groups': groups,
                    'group_ids': analysis.get('groupIds', []),
                    'statistical_analysis': {
                        'statisticalMethod': analysis.get('statisticalMethod'),
                        'pValue': analysis.get('pValue'),
                        'pValueComment': analysis.get('pValueComment'),
                        'ciLowerLimit': analysis.get('ciLowerLimit'),
                        'ciUpperLimit': analysis.get('ciUpperLimit'),
                        'ciPercentileNtiles': analysis.get('ciNumSides'),
                        'testedNonInferiority': analysis.get('testedNonInferiority'),
                        'nonInferiorityType': analysis.get('nonInferiorityType')
                    },
                    'trial_summary': description.get('briefSummary', '')
                }
                outcome_analyses.append(outcome_analysis)
        
        return {'outcome_analyses': outcome_analyses}
    
    def _process_analysis(self, analysis: Dict) -> Dict:
        """
        Process a single statistical analysis.
        
        Uses two-step processing:
        1. Generate comparative triplet
        2. Validate statistical significance
        """
        group_ids = analysis.get('group_ids', [])
        
        # Determine analysis type
        if len(group_ids) == 1:
            analysis_type = 'SINGLE_GROUP'
        elif len(group_ids) == 2:
            analysis_type = 'TWO_GROUP'
        else:
            # More than 2 groups - skip for now
            return None
        
        # Filter groups to only those in the comparison
        groups = [g for g in analysis.get('groups', []) if g['group_id'] in group_ids]
        
        # Prepare Step 1 data
        step1_data = {
            'trial_summary': analysis.get('trial_summary', ''),
            'outcome': {
                'title': analysis.get('outcome_title'),
                'description': analysis.get('outcome_description'),
                'type': analysis.get('outcome_type'),
                'unit_of_measure': analysis.get('unit_of_measure'),
                'time_frame': analysis.get('time_frame'),
                'param_type': analysis.get('param_type')
            },
            'groups': groups
        }
        
        # Check for non-inferiority
        is_non_inferiority = analysis.get('statistical_analysis', {}).get('testedNonInferiority', False)
        
        # Step 1: Generate triplet
        prompt1 = get_step1_prompt(step1_data, analysis_type, is_non_inferiority)
        response1 = self.call_llm(prompt1)
        result1 = parse_step1_response(response1)
        
        if 'error' in result1:
            return None
        
        # Step 2: Validate significance
        step2_data = {
            'statistical_analysis': analysis.get('statistical_analysis', {})
        }
        
        prompt2 = get_step2_prompt(result1.get('triplet', {}), step2_data, step1_data)
        response2 = self.call_llm(prompt2)
        result2 = parse_step2_response(response2)
        
        # Build conclusion
        conclusion = {
            'outcome_title': analysis.get('outcome_title'),
            'outcome_type': analysis.get('outcome_type'),
            'analysis_type': 'comparative' if len(group_ids) > 1 else 'non_comparative',
            'triplet': result1.get('triplet', {}),
            'logic_trace': result1.get('logic_trace', {}),
            'statistical_significance': result2.get('is_statistically_significant'),
            'statistical_conclusion': result2.get('statistical_conclusion'),
            'group_ids': group_ids,
            'p_value': analysis.get('statistical_analysis', {}).get('pValue')
        }
        
        return conclusion


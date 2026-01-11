"""
Task 6: Mapping to Disease-Level Clinical Impact

Bridges intervention-outcome findings to disease/symptom-level effects through 
context-aware inference. This is NOT a simple mapping to trial conditions, but 
an inference step that identifies which diseases/symptoms are ACTUALLY affected 
by the observed outcome change.

Pipeline:
1. Select statistically significant intervention-outcome conclusions as candidates
2. Enrich with outcome definitions, summary statistics, and study objectives
3. LLM-based inference to determine:
   - What disease/symptom/condition is ACTUALLY affected by this outcome change
   - Whether the effect is clinically meaningful
   - The nature of the intervention-disease relationship

Key insight: The target disease is NOT necessarily the trial's registered condition.
For example:
- A diabetes trial measuring "HbA1c reduction" affects "Type 2 Diabetes Mellitus"
- But measuring "cardiovascular events" might affect "Cardiovascular Disease"
- Measuring "quality of life" might affect "Depression" or "Chronic Pain"

The model must INFER the appropriate disease/symptom based on clinical semantics.
"""

from typing import Dict, List, Any, Optional
import json
import logging

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Step 1: Infer what disease/symptom is affected by this outcome
DISEASE_INFERENCE_PROMPT_PART1 = """
# Role
You are a Clinical Evidence Inference Engine specialized in linking intervention effects to disease-level impacts.

# Task
Given a statistically significant intervention-outcome conclusion from a clinical trial, INFER which disease(s), symptom(s), or clinical condition(s) are ACTUALLY affected by this outcome change.

**CRITICAL**: The target disease is NOT necessarily the trial's registered condition. You must infer based on clinical semantics.

# Context Understanding
- The outcome measure represents a specific clinical measurement or endpoint
- A change in this outcome implies an effect on some underlying disease/symptom
- You need to determine WHAT disease/symptom/condition is affected by this outcome change

# Examples of Inference Logic
| Outcome Measure | Affected Disease/Symptom |
|-----------------|--------------------------|
| HbA1c reduction | Type 2 Diabetes Mellitus, Glycemic Control |
| Pain score (VAS) | Chronic Pain, condition-specific pain (e.g., Osteoarthritis) |
| MADRS score | Major Depressive Disorder, Depression |
| Blood pressure | Hypertension, Cardiovascular Risk |
| Tumor response rate | Cancer (specific type) |
| FEV1 improvement | Chronic Obstructive Pulmonary Disease, Asthma |
| Quality of life score | May affect multiple conditions depending on context |
| Mortality rate | The primary disease being treated |

# Input Information
## Intervention
{intervention}

## Outcome Measure
- Title: {outcome_title}
- Description: {outcome_description}
- Type: {outcome_type}
- Unit: {unit_of_measure}
- Time Frame: {time_frame}

## Statistical Result
- Effect Direction: {effect_direction}
- Statistical Significance: {significance}
- P-value: {p_value}
- Confidence Interval: {confidence_interval}

## Trial Context
- Registered Conditions: {registered_conditions}
- Study Objective: {study_objective}

# Instructions
1. Analyze the CLINICAL MEANING of the outcome measure
2. Consider what disease/symptom/condition would be affected by a change in this outcome
3. The inferred disease may be:
   - The same as the registered condition
   - A related but different condition (e.g., complication, comorbidity)
   - A symptom of the condition
   - A broader disease category
4. Do NOT simply copy the registered condition - INFER based on clinical logic
5. Consider multiple affected conditions if applicable
"""

DISEASE_INFERENCE_PROMPT_PART2 = """
# Output Format
Return a valid JSON object:

{{
    "inferred_diseases": [
        {{
            "disease_name": "Standardized disease/symptom name",
            "disease_type": "disease" | "symptom" | "syndrome" | "biomarker" | "risk_factor",
            "inference_confidence": "HIGH" | "MEDIUM" | "LOW",
            "inference_reasoning": "Why this outcome change affects this disease/symptom"
        }}
    ],
    "primary_affected_condition": "The MAIN disease/symptom affected",
    "outcome_clinical_domain": "The clinical domain (e.g., glycemic control, cardiovascular, neurological)",
    "is_same_as_registered": true | false,
    "clinical_interpretation": "Brief interpretation of what this outcome change means clinically"
}}

# Important Notes
- Use standardized medical terminology (e.g., "Type 2 Diabetes Mellitus" not "T2DM")
- Do NOT use abbreviations
- If unsure, set inference_confidence to "LOW"
- Multiple diseases can be affected by a single outcome

Return ONLY the JSON object.
"""

# Step 2: Evaluate clinical meaningfulness and determine relation type
CLINICAL_MEANINGFULNESS_PROMPT = """
# Role
You are a Clinical Evidence Evaluator assessing intervention-disease relationships.

# Task
Given the inferred disease/symptom and the intervention's effect on the outcome, determine:
1. Whether the effect is CLINICALLY MEANINGFUL (not just statistically significant)
2. The appropriate INTERVENTION-DISEASE RELATION type

# Input
## Intervention
{intervention}

## Inferred Affected Disease/Symptom
{inferred_disease}

## Outcome Evidence
- Outcome: {outcome_title}
- Effect Direction: {effect_direction}
- P-value: {p_value}
- Effect Magnitude: {effect_magnitude}

## Clinical Context
- Clinical Domain: {clinical_domain}
- Study Objective: {study_objective}

# Clinical Meaningfulness Criteria
A clinically meaningful effect should:
- Have a relevant effect SIZE (not just statistical significance)
- Align with the trial's therapeutic intent
- Represent a benefit that would matter to patients
- Be consistent with established clinical knowledge

# Intervention-Disease Relation Types
Choose the MOST appropriate relation:
- **treat**: The intervention treats or cures the disease (primary therapeutic effect)
- **improve**: The intervention improves symptoms or disease manifestations
- **alleviate**: The intervention alleviates/relieves symptoms temporarily
- **prevent**: The intervention prevents the disease or its progression
- **reduce_risk**: The intervention reduces risk of the condition
- **no_clinical_effect**: Statistically significant but NOT clinically meaningful
- **worsen**: The intervention worsens the condition (adverse effect)

# Output Format
Return a valid JSON object:

{{
    "is_clinically_meaningful": true | false,
    "intervention_disease_relation": "treat" | "improve" | "alleviate" | "prevent" | "reduce_risk" | "no_clinical_effect" | "worsen",
    "effect_interpretation": {{
        "direction": "beneficial" | "harmful" | "neutral",
        "magnitude": "large" | "moderate" | "small" | "minimal",
        "clinical_relevance": "Brief explanation of clinical relevance"
    }},
    "evidence_quality": {{
        "strength": "STRONG" | "MODERATE" | "WEAK",
        "limitations": ["list of limitations if any"]
    }},
    "final_triplet": {{
        "head_entity": "Intervention name (standardized)",
        "relation": "The relation type",
        "tail_entity": "Disease/symptom name (standardized)",
        "relation_qualifier": "Additional context if needed"
    }},
    "reasoning": "Comprehensive explanation of the assessment"
}}

Return ONLY the JSON object.
"""


class Task6DiseaseMapping(BaseTaskHandler):
    """
    Task 6: Mapping to Disease-Level Clinical Impact
    
    Links intervention-outcome findings to disease/symptom-level effects 
    through context-aware inference pipeline.
    
    Key Features:
    - Does NOT simply map to registered conditions
    - INFERS which diseases/symptoms are affected based on outcome semantics
    - Two-step LLM reasoning: (1) Disease inference, (2) Clinical meaningfulness
    - Produces intervention-disease triplets grounded in clinical context
    """
    
    @property
    def task_name(self) -> str:
        return "task6_disease_mapping"
    
    def execute(
        self, 
        trial_data: Any, 
        task5_results: Dict = None,
        **kwargs
    ) -> TaskResult:
        """
        Execute disease-level inference for a trial.
        
        Args:
            trial_data: TrialData object
            task5_results: Results from Task 5 (statistical conclusions)
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with inferred intervention-disease relations
        """
        try:
            # Get conclusions from Task 5
            if task5_results is None:
                return self._create_error_result(['Task 5 results required for disease mapping'])
            
            conclusions = task5_results.get('conclusions', [])
            
            # Filter to statistically significant conclusions
            significant_conclusions = [
                c for c in conclusions 
                if c.get('statistical_significance', False)
            ]
            
            if not significant_conclusions:
                return self._create_success_result({
                    'disease_relations': [],
                    'kg_triples': [],
                    'message': 'No significant conclusions to map'
                })
            
            # Prepare trial context
            context = self.prepare_input(trial_data)
            
            # Process each significant conclusion through two-step inference
            disease_relations = []
            all_triples = []
            errors = []
            inference_details = []
            
            for conclusion in significant_conclusions:
                try:
                    # Step 1: Infer affected disease/symptom
                    inferred = self._step1_infer_disease(conclusion, context)
                    
                    if not inferred or 'error' in inferred:
                        errors.append(f"Disease inference failed for {conclusion.get('outcome_title')}")
                        continue
                    
                    # Step 2: Evaluate clinical meaningfulness for each inferred disease
                    for disease_info in inferred.get('inferred_diseases', []):
                        result = self._step2_evaluate_meaningfulness(
                            conclusion, disease_info, inferred, context
                        )
                        
                        if result and result.get('is_clinically_meaningful', False):
                            # Build complete relation record
                            relation = self._build_relation_record(
                                conclusion, disease_info, inferred, result, context
                            )
                            disease_relations.append(relation)
                            
                            # Generate KG triples
                            triples = self._generate_kg_triples(relation)
                            all_triples.extend(triples)
                    
                    # Store inference details for transparency
                    inference_details.append({
                        'outcome': conclusion.get('outcome_title'),
                        'inferred_conditions': inferred.get('inferred_diseases', []),
                        'is_same_as_registered': inferred.get('is_same_as_registered'),
                        'clinical_interpretation': inferred.get('clinical_interpretation')
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process conclusion: {e}")
                    errors.append(str(e))
            
            return self._create_success_result({
                'disease_relations': disease_relations,
                'kg_triples': all_triples,
                'inference_details': inference_details,
                'summary': {
                    'total_significant_conclusions': len(significant_conclusions),
                    'clinically_meaningful_relations': len(disease_relations),
                    'unique_inferred_diseases': len(set(
                        r.get('inferred_disease') for r in disease_relations
                    )),
                    'same_as_registered_count': sum(
                        1 for d in inference_details if d.get('is_same_as_registered')
                    ),
                    'novel_inference_count': sum(
                        1 for d in inference_details if not d.get('is_same_as_registered')
                    )
                },
                'errors': errors
            })
            
        except Exception as e:
            logger.error(f"Task 6 failed: {e}")
            return self._create_error_result([str(e)])
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare trial context for disease inference.
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with trial context information
        """
        conditions = getattr(trial_data, 'conditions', {})
        description = getattr(trial_data, 'description', {})
        identification = getattr(trial_data, 'identification', {})
        
        return {
            'nct_id': trial_data.nct_id,
            'registered_conditions': conditions.get('conditions', []),
            'keywords': conditions.get('keywords', []),
            'brief_summary': description.get('briefSummary', ''),
            'detailed_description': description.get('detailedDescription', ''),
            'title': identification.get('officialTitle', identification.get('briefTitle', ''))
        }
    
    def _step1_infer_disease(self, conclusion: Dict, context: Dict) -> Optional[Dict]:
        """
        Step 1: Infer which disease/symptom is affected by the outcome change.
        
        This is the KEY inference step - we don't just use registered conditions,
        but infer based on clinical semantics of the outcome.
        
        Args:
            conclusion: Statistical conclusion from Task 5
            context: Trial context
            
        Returns:
            Inference result with inferred diseases
        """
        triplet = conclusion.get('triplet', {})
        stat_analysis = conclusion.get('statistical_analysis', {}) or {}
        
        # Extract intervention
        intervention = triplet.get('head_entity_name', '')
        if not intervention:
            # Try alternative field names
            intervention = triplet.get('head_entity', '') or 'Unknown intervention'
        
        # Extract effect information
        effect_direction = triplet.get('relationship_context', '')
        comparative_relation = triplet.get('comparative_relation', '')
        
        # Get p-value and CI
        p_value = conclusion.get('p_value', 'N/A')
        ci_lower = stat_analysis.get('ciLowerLimit', '')
        ci_upper = stat_analysis.get('ciUpperLimit', '')
        ci_str = f"[{ci_lower}, {ci_upper}]" if ci_lower and ci_upper else "N/A"
        
        # Build prompt
        prompt1 = DISEASE_INFERENCE_PROMPT_PART1.format(
            intervention=intervention,
            outcome_title=conclusion.get('outcome_title', 'Unknown'),
            outcome_description=conclusion.get('outcome_description', '')[:500] if conclusion.get('outcome_description') else 'N/A',
            outcome_type=conclusion.get('outcome_type', 'N/A'),
            unit_of_measure=conclusion.get('unit_of_measure', 'N/A'),
            time_frame=conclusion.get('time_frame', 'N/A'),
            effect_direction=f"{comparative_relation} {effect_direction}".strip() or 'N/A',
            significance=conclusion.get('statistical_conclusion', 'Statistically significant'),
            p_value=p_value,
            confidence_interval=ci_str,
            registered_conditions=json.dumps(context.get('registered_conditions', []), ensure_ascii=False),
            study_objective=context.get('brief_summary', '')[:800] or 'N/A'
        )
        
        full_prompt = prompt1 + "\n" + DISEASE_INFERENCE_PROMPT_PART2
        
        # Call LLM
        response = self.call_llm(full_prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def _step2_evaluate_meaningfulness(
        self, 
        conclusion: Dict, 
        disease_info: Dict,
        inference_result: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """
        Step 2: Evaluate clinical meaningfulness and determine relation type.
        
        Args:
            conclusion: Statistical conclusion from Task 5
            disease_info: Single inferred disease from Step 1
            inference_result: Full inference result from Step 1
            context: Trial context
            
        Returns:
            Clinical meaningfulness evaluation result
        """
        triplet = conclusion.get('triplet', {})
        
        # Extract intervention
        intervention = triplet.get('head_entity_name', '') or triplet.get('head_entity', '') or 'Unknown'
        
        # Effect information
        effect_direction = triplet.get('relationship_context', '')
        comparative_relation = triplet.get('comparative_relation', '')
        
        # Build prompt
        prompt = CLINICAL_MEANINGFULNESS_PROMPT.format(
            intervention=intervention,
            inferred_disease=disease_info.get('disease_name', 'Unknown'),
            outcome_title=conclusion.get('outcome_title', 'Unknown'),
            effect_direction=f"{comparative_relation} {effect_direction}".strip() or 'N/A',
            p_value=conclusion.get('p_value', 'N/A'),
            effect_magnitude=conclusion.get('statistical_conclusion', 'N/A'),
            clinical_domain=inference_result.get('outcome_clinical_domain', 'N/A'),
            study_objective=context.get('brief_summary', '')[:500] or 'N/A'
        )
        
        # Call LLM
        response = self.call_llm(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def _build_relation_record(
        self,
        conclusion: Dict,
        disease_info: Dict,
        inference_result: Dict,
        meaningfulness_result: Dict,
        context: Dict
    ) -> Dict:
        """Build a complete relation record combining all inference results."""
        triplet = conclusion.get('triplet', {})
        final_triplet = meaningfulness_result.get('final_triplet', {})
        
        return {
            'intervention': final_triplet.get('head_entity') or triplet.get('head_entity_name', ''),
            'inferred_disease': disease_info.get('disease_name', ''),
            'disease_type': disease_info.get('disease_type', 'disease'),
            'relation': final_triplet.get('relation') or meaningfulness_result.get('intervention_disease_relation', ''),
            'relation_qualifier': final_triplet.get('relation_qualifier', ''),
            
            # Inference metadata
            'inference_confidence': disease_info.get('inference_confidence', 'MEDIUM'),
            'is_same_as_registered': inference_result.get('is_same_as_registered', False),
            'clinical_domain': inference_result.get('outcome_clinical_domain', ''),
            
            # Clinical meaningfulness
            'is_clinically_meaningful': meaningfulness_result.get('is_clinically_meaningful', False),
            'effect_interpretation': meaningfulness_result.get('effect_interpretation', {}),
            'evidence_quality': meaningfulness_result.get('evidence_quality', {}),
            
            # Source information
            'source_outcome': conclusion.get('outcome_title', ''),
            'source_conclusion': conclusion,
            'reasoning': {
                'disease_inference': disease_info.get('inference_reasoning', ''),
                'clinical_assessment': meaningfulness_result.get('reasoning', '')
            },
            
            # Context
            'registered_conditions': context.get('registered_conditions', []),
            'nct_id': context.get('nct_id', '')
        }
    
    def _generate_kg_triples(self, relation: Dict) -> List[Dict]:
        """Generate knowledge graph triples from a relation record."""
        triples = []
        
        intervention = relation.get('intervention', '')
        disease = relation.get('inferred_disease', '')
        relation_type = relation.get('relation', '')
        nct_id = relation.get('nct_id', '')
        
        if not all([intervention, disease, relation_type]):
            return triples
        
        # Main intervention-disease relation triple
        triples.append({
            'head': intervention,
            'relation': relation_type,
            'tail': disease,
            'head_type': 'Intervention',
            'tail_type': relation.get('disease_type', 'Disease').capitalize(),
            'source_trial': nct_id,
            'evidence_strength': relation.get('evidence_quality', {}).get('strength', 'MODERATE'),
            'inference_confidence': relation.get('inference_confidence', 'MEDIUM'),
            'is_inferred': not relation.get('is_same_as_registered', True),
            'attributes': {
                'clinical_domain': relation.get('clinical_domain', ''),
                'source_outcome': relation.get('source_outcome', ''),
                'effect_direction': relation.get('effect_interpretation', {}).get('direction', ''),
                'effect_magnitude': relation.get('effect_interpretation', {}).get('magnitude', '')
            }
        })
        
        # Link intervention to trial
        triples.append({
            'head': nct_id,
            'relation': 'has_intervention_disease_finding',
            'tail': f"{intervention}_{relation_type}_{disease}",
            'head_type': 'Trial',
            'tail_type': 'Finding'
        })
        
        return triples
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON."""
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
        
        return {
            'error': 'Failed to parse response',
            'raw_response': response[:500] if response else ''
        }


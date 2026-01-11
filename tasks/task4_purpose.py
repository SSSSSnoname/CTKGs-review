"""
Task 4: Inferring Study Purpose

Infers study purpose by jointly analyzing:
- Trial titles
- Descriptions (briefSummary)
- Interventions
- Conditions
- Design metadata

Outputs structured triplets linking interventions (head entities) to 
diseases/symptoms/endpoints (tail entities) with standardized purpose relations.
"""

from typing import Dict, List, Any
import json
import logging

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


def generate_purpose_prompts(interventions: List[Dict], conditions: List[str]) -> tuple:
    """
    Generate two-part prompts for purpose inference with dynamic entity fields.
    
    Args:
        interventions: List of intervention dictionaries
        conditions: List of condition strings
        
    Returns:
        Tuple of (prompt1, prompt2, initial_triplets)
    """
    # Extract intervention names as head entities
    head_entities = []
    for intervention in interventions:
        name = intervention.get('name', '')
        if name:
            head_entities.append(name)
    
    entity_count = len(head_entities) if head_entities else 1
    entity_word = "intervention" if entity_count == 1 else "interventions"
    
    prompt1 = f"""
# Role
You are a Clinical Trial Purpose Inference Engine specialized in extracting intervention-disease relationships.

# Task
Given the briefSummary of the trial, determine the PURPOSE relationship between the given {entity_word} (head entities) and their target diseases/symptoms/endpoints (tail entities).

# Instructions
- **Correct and update the 'head_entity' fields** based on the information in the trial summary
  - Ensure that head entities are INTERVENTIONS, not diseases or symptoms
  - If the head entity is a combination drug or therapy, specify the COMPLETE combination
  - Do NOT use abbreviations
  - Do NOT output erroneous or redundant head entities

- **Correct and update the 'tail_entity' fields** based on the information in the trial summary
  - Tail entities should be diseases, symptoms, or clinical endpoints
  - Extract tail entities DIRECTLY from the trial summary
  - If your tail entity is empty, the answer should be "no"

- **Correct and update the 'purpose' field** based on the information in the trial summary
  - Use ONLY these standardized purpose options:
    * **treat**: The intervention is used to treat or cure the condition
    * **improve**: The intervention aims to improve symptoms or outcomes
    * **support**: The intervention provides supportive care
    * **decrease**: The intervention aims to decrease/reduce something (e.g., side effects, risk)
    * **increase**: The intervention aims to increase something (e.g., survival, response rate)

# Provided Head Entities (Interventions)
{json.dumps(head_entities, indent=2) if head_entities else '["Unknown intervention"]'}

# Provided Conditions
{json.dumps(conditions, indent=2) if conditions else '["Unknown condition"]'}
"""

    # Build dynamic entity fields for output format
    entity_fields = ""
    for idx in range(entity_count):
        entity_fields += f'    "head_entity_{idx+1}": "Corrected head entity {idx+1}",\n'
    
    prompt2 = f"""
# Output Format
Please respond with a valid JSON object in the following format:

{{
    "answer": "yes" or "no",
    "reason": "Provide a brief explanation of the purpose relationship",
{entity_fields}    "purpose": "treat" | "improve" | "support" | "decrease" | "increase",
    "tail_entity": ["Corrected tail entity (disease/symptom/endpoint)", ...]
}}

# Important Notes
- If the purpose relationship cannot be determined, set "answer" to "no"
- Each head_entity should be a specific intervention (drug, therapy, procedure, etc.)
- Each tail_entity should be a disease, symptom, or clinical endpoint
- Use standardized medical terminology for all entities

Return ONLY the JSON object, no additional text.
"""

    # Prepare initial triplets structure
    triplets = {}
    for idx, entity in enumerate(head_entities if head_entities else ["Unknown"]):
        triplets[f'head_entity_{idx+1}'] = entity
    triplets['purpose'] = "treat"  # Default purpose
    triplets['tail_entity'] = conditions if conditions else []
    
    return prompt1, prompt2, triplets


# Fallback simple prompt for cases with missing data
PURPOSE_PROMPT_SIMPLE = """
# Role
You are a Clinical Trial Purpose Inference Engine.

# Task
Infer the primary purpose relationship between interventions and target conditions.

# Instructions
1. Identify the intervention(s) being studied (head entities)
2. Identify the target disease(s)/symptom(s)/endpoint(s) (tail entities)  
3. Determine the purpose relationship using ONLY these options:
   - treat: The intervention treats or cures the condition
   - improve: The intervention improves symptoms or outcomes
   - support: The intervention provides supportive care
   - decrease: The intervention decreases/reduces something
   - increase: The intervention increases something

# Input Data
Title: {title}
Brief Summary: {summary}
Conditions: {conditions}
Interventions: {interventions}

# Output Format
{{
    "answer": "yes" or "no",
    "reason": "Brief explanation",
    "head_entities": ["intervention 1", "intervention 2", ...],
    "purpose": "treat" | "improve" | "support" | "decrease" | "increase",
    "tail_entities": ["disease/symptom/endpoint 1", ...],
    "confidence": "HIGH" | "MEDIUM" | "LOW"
}}

Return ONLY the JSON object.
"""


class Task4PurposeInference(BaseTaskHandler):
    """
    Task 4: Inferring Study Purpose
    
    Recovers study intent when registry-level Purpose labels are
    missing, inconsistent, or overly generic.
    
    Outputs structured triplets:
    - head_entity: Intervention (drug, therapy, procedure)
    - purpose: treat / improve / support / decrease / increase
    - tail_entity: Disease, symptom, or clinical endpoint
    """
    
    # Valid purpose options
    VALID_PURPOSES = ['treat', 'improve', 'support', 'decrease', 'increase']
    
    @property
    def task_name(self) -> str:
        return "task4_purpose_inference"
    
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute purpose inference for a trial.
        
        Args:
            trial_data: TrialData object
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with inferred purpose triplets
        """
        try:
            # Prepare input data
            inputs = self.prepare_input(trial_data)
            
            existing_purpose = inputs.get('design_purpose', '')
            interventions = inputs.get('interventions', [])
            conditions = inputs.get('conditions', [])
            summary = inputs.get('summary', '')
            
            # Generate dynamic prompts based on interventions
            prompt1, prompt2, initial_triplets = generate_purpose_prompts(
                interventions, conditions
            )
            
            # Build complete prompt with summary context
            full_prompt = f"""
# Brief Summary
{summary[:3000] if summary else 'No summary available'}

# Title
{inputs.get('title', 'N/A')}

{prompt1}
{prompt2}
"""
            
            # Call LLM
            response = self.call_llm(full_prompt)
            
            # Parse response
            inferred = self._parse_response(response)
            
            # Validate and normalize the purpose
            inferred = self._normalize_purpose_response(inferred, initial_triplets)
            
            # Add original purpose for comparison
            inferred['original_purpose'] = existing_purpose
            inferred['nct_id'] = trial_data.nct_id
            
            # Generate knowledge graph triples
            inferred['kg_triples'] = self._generate_kg_triples(
                trial_data.nct_id, inferred
            )
            
            return self._create_success_result({
                'inferred_purpose': inferred
            })
            
        except Exception as e:
            logger.error(f"Task 4 failed: {e}")
            return self._create_error_result([str(e)])
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare trial data for purpose inference.
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary with trial metadata
        """
        identification = getattr(trial_data, 'identification', {})
        description = getattr(trial_data, 'description', {})
        conditions = getattr(trial_data, 'conditions', {})
        design = getattr(trial_data, 'design', {})
        arms = getattr(trial_data, 'arms_interventions', {})
        
        # Extract intervention names
        interventions = []
        for intervention in arms.get('interventions', []):
            interventions.append({
                'name': intervention.get('name'),
                'type': intervention.get('type'),
                'description': intervention.get('description', '')[:200]  # Truncate
            })
        
        return {
            'title': identification.get('officialTitle', identification.get('briefTitle', '')),
            'summary': description.get('briefSummary', ''),
            'conditions': conditions.get('conditions', []),
            'interventions': interventions,
            'design_purpose': design.get('designInfo', {}).get('primaryPurpose', '')
        }
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract inferred purpose."""
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse purpose JSON: {e}")
        
        return {
            'error': 'Failed to parse response',
            'raw_response': response[:500]
        }
    
    def _normalize_purpose_response(self, inferred: Dict, initial_triplets: Dict) -> Dict:
        """Normalize and validate the purpose response."""
        # Ensure 'answer' field exists
        if 'answer' not in inferred:
            inferred['answer'] = 'yes' if inferred.get('purpose') else 'no'
        
        # Validate purpose is one of the allowed options
        purpose = inferred.get('purpose', '').lower()
        if purpose not in self.VALID_PURPOSES:
            # Try to map common alternatives
            purpose_map = {
                'treatment': 'treat',
                'treating': 'treat',
                'cure': 'treat',
                'improvement': 'improve',
                'improving': 'improve',
                'reduce': 'decrease',
                'reduction': 'decrease',
                'lower': 'decrease',
                'raise': 'increase',
                'enhance': 'increase',
                'supportive': 'support'
            }
            inferred['purpose'] = purpose_map.get(purpose, 'treat')
        else:
            inferred['purpose'] = purpose
        
        # Collect all head entities
        head_entities = []
        for key, value in inferred.items():
            if key.startswith('head_entity_') and value:
                head_entities.append(value)
        inferred['head_entities'] = head_entities if head_entities else initial_triplets.get('head_entities', [])
        
        # Ensure tail_entity is a list
        tail = inferred.get('tail_entity', [])
        if isinstance(tail, str):
            inferred['tail_entity'] = [tail] if tail else []
        elif not isinstance(tail, list):
            inferred['tail_entity'] = []
        
        return inferred
    
    def _generate_kg_triples(self, nct_id: str, inferred: Dict) -> List[Dict]:
        """Generate knowledge graph triples from inferred purpose."""
        triples = []
        
        purpose = inferred.get('purpose', 'treat')
        head_entities = inferred.get('head_entities', [])
        tail_entities = inferred.get('tail_entity', [])
        
        # Generate triplets for each head-tail combination
        for head in head_entities:
            if not head:
                continue
            for tail in tail_entities:
                if not tail:
                    continue
                triples.append({
                    'head': head,
                    'relation': purpose,
                    'tail': tail,
                    'head_type': 'Intervention',
                    'tail_type': 'Disease/Symptom/Endpoint',
                    'source_trial': nct_id
                })
        
        # Also link interventions to the trial
        for head in head_entities:
            if head:
                triples.append({
                    'head': nct_id,
                    'relation': 'studies_intervention',
                    'tail': head,
                    'head_type': 'Trial',
                    'tail_type': 'Intervention'
                })
        
        return triples


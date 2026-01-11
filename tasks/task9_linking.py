"""
Task 9: Post-Processing and Ontology Integration

Multi-stage cascade entity-linking framework:

Stage I: Federated ontology retrieval via SRI Name Resolution API
        Re-scored using SapBERT + string-based matching
        
Stage II-III: Retrieve-and-rerank with SapBERT bi-encoder + FAISS
              Cross-encoder reranking for fine-grained disambiguation
              Integrates UMLS and biomedical knowledge bases
              
Stage IV: LLM-based semantic normalization for composite mentions

Residual unlinked entities consolidated via embedding-based clustering.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


class Task9EntityLinking(BaseTaskHandler):
    """
    Task 9: Post-Processing and Ontology Integration
    
    Normalizes heterogeneous biomedical mentions through a
    multi-stage cascade entity-linking framework.
    """
    
    @property
    def task_name(self) -> str:
        return "task9_entity_linking"
    
    def execute(
        self, 
        trial_data: Any,
        task_results: Dict = None,
        linking_config: Dict = None,
        **kwargs
    ) -> TaskResult:
        """
        Execute entity linking for extracted entities.
        
        Args:
            trial_data: TrialData object
            task_results: Results from previous tasks
            linking_config: Configuration for entity linking
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with linked entities
        """
        try:
            task_results = task_results or {}
            linking_config = linking_config or {}
            
            # Collect all entities to link
            entities = self._collect_entities(trial_data, task_results)
            
            if not entities:
                return self._create_success_result({
                    'linked_entities': [],
                    'message': 'No entities to link'
                })
            
            # Execute multi-stage linking
            linked_entities = []
            unlinked_entities = []
            
            for entity in entities:
                try:
                    result = self._link_entity(entity, linking_config)
                    if result.get('linked'):
                        linked_entities.append(result)
                    else:
                        unlinked_entities.append(result)
                except Exception as e:
                    logger.warning(f"Failed to link entity '{entity['text']}': {e}")
                    unlinked_entities.append({
                        'original': entity,
                        'linked': False,
                        'error': str(e)
                    })
            
            # Cluster residual unlinked entities
            clusters = self._cluster_unlinked(unlinked_entities)
            
            return self._create_success_result({
                'linked_entities': linked_entities,
                'unlinked_entities': unlinked_entities,
                'clusters': clusters,
                'total_entities': len(entities),
                'linked_count': len(linked_entities),
                'unlinked_count': len(unlinked_entities),
                'cluster_count': len(clusters)
            })
            
        except Exception as e:
            logger.error(f"Task 9 failed: {e}")
            return self._create_error_result([str(e)])
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """Prepare trial data for entity linking."""
        return {'nct_id': trial_data.nct_id}
    
    def _collect_entities(
        self, 
        trial_data: Any, 
        task_results: Dict
    ) -> List[Dict]:
        """
        Collect all entities from trial data and task results.
        
        Entities include:
        - Conditions/diseases
        - Interventions/drugs
        - Outcomes
        - Adverse events
        """
        entities = []
        
        # Conditions
        conditions = getattr(trial_data, 'conditions', {})
        for condition in conditions.get('conditions', []):
            entities.append({
                'text': condition,
                'type': 'condition',
                'source': 'trial_data'
            })
        
        # Interventions from Task 2
        task2 = task_results.get('task2_intervention_profiling', {})
        for arm in task2.get('profiled_interventions', []):
            for intervention in arm.get('interventions', []):
                name = intervention.get('name')
                if name and name.lower() != 'placebo':
                    entities.append({
                        'text': name,
                        'type': 'intervention',
                        'source': 'task2',
                        'intervention_type': intervention.get('type')
                    })
        
        # Outcomes from Task 1
        task1 = task_results.get('task1_outcome_standardization', {})
        for outcome in task1.get('standardized_outcomes', []):
            for normalized in outcome.get('standardized', []):
                core = normalized.get('core_measurement')
                if core:
                    entities.append({
                        'text': core,
                        'type': 'outcome',
                        'source': 'task1'
                    })
        
        # Adverse events
        ae_module = getattr(trial_data, 'adverse_events', {})
        for event in ae_module.get('seriousEvents', []):
            term = event.get('term')
            if term:
                entities.append({
                    'text': term,
                    'type': 'adverse_event',
                    'source': 'trial_data'
                })
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['text'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _link_entity(self, entity: Dict, config: Dict) -> Dict:
        """
        Link a single entity through the multi-stage cascade.
        
        Stages:
        1. SRI Name Resolution with hybrid reranking
        2. Local ontology search (SapBERT + FAISS)
        3. UMLS search
        4. LLM-based normalization for failures
        """
        text = entity['text']
        entity_type = entity['type']
        
        # Stage 1: SRI Name Resolution
        stage1_result = self._stage1_sri_resolution(text, entity_type, config)
        if stage1_result.get('score', 0) >= config.get('stage1_threshold', 0.8):
            return {
                'original': entity,
                'linked': True,
                'stage': 'stage1_sri',
                'result': stage1_result
            }
        
        # Stage 2: Local ontology search
        stage2_result = self._stage2_ontology_search(text, entity_type, config)
        if stage2_result.get('score', 0) >= config.get('stage2_threshold', 0.9):
            return {
                'original': entity,
                'linked': True,
                'stage': 'stage2_ontology',
                'result': stage2_result
            }
        
        # Stage 3: UMLS search
        stage3_result = self._stage3_umls_search(text, entity_type, config)
        if stage3_result.get('score', 0) >= config.get('stage3_threshold', 0.92):
            return {
                'original': entity,
                'linked': True,
                'stage': 'stage3_umls',
                'result': stage3_result
            }
        
        # Stage 4: LLM-based normalization
        stage4_result = self._stage4_llm_normalization(text, entity_type, config)
        if stage4_result.get('normalized'):
            # Recursively try to link normalized terms
            for normalized_term in stage4_result.get('split_concepts', []):
                relink = self._link_entity(
                    {'text': normalized_term, 'type': entity_type},
                    config
                )
                if relink.get('linked'):
                    return {
                        'original': entity,
                        'linked': True,
                        'stage': 'stage4_llm',
                        'normalized_to': normalized_term,
                        'result': relink.get('result')
                    }
        
        # Failed to link
        return {
            'original': entity,
            'linked': False,
            'best_candidate': stage1_result or stage2_result or stage3_result
        }
    
    def _stage1_sri_resolution(
        self, 
        text: str, 
        entity_type: str, 
        config: Dict
    ) -> Dict:
        """
        Stage 1: SRI Name Resolution API with hybrid reranking.
        
        Uses SapBERT embeddings + string-based matching for reranking.
        """
        # NOTE: This is a placeholder implementation
        # In production, integrate with actual SRI API and SapBERT
        
        try:
            # Import the actual implementation if available
            from ...data_loader.entity_linking import sri_search
            return sri_search(text, entity_type)
        except ImportError:
            logger.debug("SRI search not available, using placeholder")
            
        # Placeholder - return empty result
        return {
            'curie': None,
            'label': None,
            'source': None,
            'score': 0.0
        }
    
    def _stage2_ontology_search(
        self, 
        text: str, 
        entity_type: str, 
        config: Dict
    ) -> Dict:
        """
        Stage 2: Local ontology search with SapBERT bi-encoder + FAISS.
        """
        # NOTE: Placeholder implementation
        try:
            from ...data_loader.entity_linking import ontology_search
            return ontology_search(text, entity_type)
        except ImportError:
            pass
        
        return {
            'entity_id': None,
            'entity_ontology': None,
            'entity_onto_term': None,
            'score': 0.0
        }
    
    def _stage3_umls_search(
        self, 
        text: str, 
        entity_type: str, 
        config: Dict
    ) -> Dict:
        """
        Stage 3: UMLS search with cross-encoder reranking.
        """
        # NOTE: Placeholder implementation
        try:
            from ...data_loader.entity_linking import umls_search
            return umls_search(text, entity_type)
        except ImportError:
            pass
        
        return {
            'cui': None,
            'preferred_name': None,
            'semantic_type': None,
            'score': 0.0
        }
    
    def _stage4_llm_normalization(
        self, 
        text: str, 
        entity_type: str, 
        config: Dict
    ) -> Dict:
        """
        Stage 4: LLM-based semantic normalization.
        
        Rewrites or decomposes composite mentions into atomic concepts.
        """
        prompt = f"""
You are a biomedical entity normalization expert.

The following entity mention could not be linked to standard ontologies:
Entity: "{text}"
Entity Type: {entity_type}

Please analyze and:
1. If the mention is a composite (multiple concepts), split into atomic concepts
2. Correct any spelling or formatting issues
3. Provide alternative names or synonyms

Output JSON:
{{
    "normalized": true/false,
    "corrected_text": "corrected version if needed",
    "split_concepts": ["list", "of", "atomic", "concepts"],
    "synonyms": ["alternative", "names"]
}}
"""
        
        try:
            response = self.call_llm(prompt)
            
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.warning(f"LLM normalization failed: {e}")
        
        return {'normalized': False}
    
    def _cluster_unlinked(self, unlinked: List[Dict]) -> List[Dict]:
        """
        Cluster residual unlinked entities using embedding-based similarity.
        
        Uses SapBERT cosine similarity to form connected components,
        then selects representative labels for each cluster.
        """
        if not unlinked:
            return []
        
        # NOTE: Placeholder implementation
        # In production, use actual SapBERT embeddings for clustering
        
        clusters = []
        
        # Simple text-based grouping as placeholder
        from collections import defaultdict
        groups = defaultdict(list)
        
        for entity in unlinked:
            original = entity.get('original', {})
            text = original.get('text', '').lower()
            # Simple first-word grouping
            first_word = text.split()[0] if text else 'unknown'
            groups[first_word].append(entity)
        
        for group_key, members in groups.items():
            if len(members) > 1:
                clusters.append({
                    'cluster_id': f"cluster_{group_key}",
                    'representative': members[0].get('original', {}).get('text'),
                    'members': [m.get('original', {}).get('text') for m in members],
                    'size': len(members)
                })
        
        return clusters


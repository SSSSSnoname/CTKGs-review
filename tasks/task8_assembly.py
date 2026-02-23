"""
Task 8: Assembling Trial- and Intervention-Centric CTKGs

Integrates standardized entities into knowledge graphs using unified triple 
representation (h, r, t).

Trial-Centric CTKG:
- Anchored on NCTID
- Links all trial-specific components through typed relations
- Includes: design, conditions, interventions, outcomes, eligibility, adverse events,
  baseline characteristics, purpose, references, sponsor information

Intervention-Centric CTKG:
- Centers on interventions as primary nodes
- Links to outcomes, diseases, adverse events, and co-therapies

Reference: Trial_level_information_3.py for trial-level information structure
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class KGTriple:
    """Representation of a knowledge graph triple."""
    head: str
    relation: str
    tail: str
    head_type: str = ""
    tail_type: str = ""
    attributes: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = {
            'head': self.head,
            'relation': self.relation,
            'tail': self.tail,
            'head_type': self.head_type,
            'tail_type': self.tail_type
        }
        if self.attributes:
            data['tail_attributes'] = self.attributes
        data['provenance'] = self.provenance
        return data


@dataclass
class TrialCentricCTKG:
    """
    Trial-centric CTKG structure following Trial_level_information_3.py schema.
    
    Includes optional version history from Task 7 (Dynamic CTKG).
    """
    nct_id: str
    trials_base_information: Dict = field(default_factory=dict)
    brief_summary: str = ""
    conditions: List[str] = field(default_factory=list)
    interventions: Dict = field(default_factory=dict)  # Structured by Task 2
    eligibility_criteria: Dict = field(default_factory=dict)  # Structured by Task 3
    outcomes: Dict = field(default_factory=dict)  # Structured by Task 1
    purpose: Dict = field(default_factory=dict)  # Structured by Task 4
    statistical_conclusions: List[Dict] = field(default_factory=list)  # From Task 5
    group_baseline: Dict = field(default_factory=dict)
    adverse_events: Dict = field(default_factory=dict)
    triples: List[KGTriple] = field(default_factory=list)
    # Task 7: Dynamic CTKG - Version history
    version_history: Dict = field(default_factory=dict)  # From Task 7
    
    def to_dict(self) -> Dict:
        return {
            'nct_id': self.nct_id,
            'trials_base_information': self.trials_base_information,
            'brief_summary': self.brief_summary,
            'conditions': self.conditions,
            'interventions': self.interventions,
            'eligibility_criteria': self.eligibility_criteria,
            'outcomes': self.outcomes,
            'purpose': self.purpose,
            'statistical_conclusions': self.statistical_conclusions,
            'group_baseline': self.group_baseline,
            'adverse_events': self.adverse_events,
            'triples': [t.to_dict() for t in self.triples],
            'version_history': self.version_history
        }


class Task8CTKGAssembly(BaseTaskHandler):
    """
    Task 8: Assembling Trial- and Intervention-Centric CTKGs
    
    Integrates all standardized entities into unified knowledge graphs
    with consistent triple representation.
    
    Uses structured outputs from:
    - Task 1: Standardized outcomes
    - Task 2: Profiled interventions
    - Task 3: Structured eligibility
    - Task 4: Inferred purpose
    - Task 5: Statistical conclusions
    - Task 6: Disease mappings
    - Task 7: Dynamic CTKG (version history)
    """
    
    @property
    def task_name(self) -> str:
        return "task8_ctkg_assembly"
    
    def execute(
        self, 
        trial_data: Any,
        task_results: Dict = None,
        ctkg_type: str = "both",
        include_dynamic: bool = True,
        **kwargs
    ) -> TaskResult:
        """
        Execute CTKG assembly for a trial.
        
        Args:
            trial_data: TrialData object
            task_results: Results from previous tasks (Task 1-7)
            ctkg_type: "trial_centric", "intervention_centric", or "both"
            include_dynamic: Whether to include Task 7 dynamic CTKG data
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with assembled CTKG (both structured data and triples)
        """
        try:
            task_results = task_results or {}
            nct_id = trial_data.nct_id
            
            trial_centric_ctkg = None
            trial_centric_triples = []
            intervention_centric_triples = []
            dynamic_ctkg_triples = []
            
            # Build trial-centric CTKG (structured + triples)
            if ctkg_type in ["trial_centric", "both"]:
                trial_centric_ctkg = self._build_trial_centric_structured(
                    trial_data, 
                    task_results
                )
                trial_centric_triples = trial_centric_ctkg.triples
                self._ensure_provenance_nct_id(trial_centric_triples, nct_id)
                
                # Integrate Task 7 dynamic CTKG if available
                if include_dynamic:
                    dynamic_data = self._integrate_dynamic_ctkg(
                        trial_centric_ctkg,
                        task_results
                    )
                    dynamic_ctkg_triples = dynamic_data.get('triples', [])
                    self._ensure_provenance_nct_id_in_dicts(dynamic_ctkg_triples, nct_id)
            
            # Build intervention-centric CTKG
            raw_task2_relations = []
            raw_task56_triples = []
            if ctkg_type in ["intervention_centric", "both"]:
                # Keep Task2/Task5/Task6 relations in original schema (no rewriting).
                raw_task2_relations = self._get_task2_raw_relations(task_results)
                raw_task56_triples = self._get_task56_raw_triples(task_results)

            # Re-bucket by head-level rule:
            # - head == NCTID => trial-level
            # - head starts with NCTID_ => intervention-level (group-level)
            trial_centric_triples, intervention_centric_triples = self._rebucket_by_head_level(
                nct_id,
                trial_centric_triples,
                intervention_centric_triples,
            )
            
            intervention_output = (
                [t.to_dict() for t in intervention_centric_triples] +
                raw_task2_relations +
                raw_task56_triples
            )

            return self._create_success_result({
                'trial_centric_ctkg': (
                    {
                        'triples': [t.to_dict() for t in trial_centric_triples],
                        'version_history': trial_centric_ctkg.version_history if trial_centric_ctkg else {}
                    } if trial_centric_ctkg else None
                ),
                'trial_centric_triples': [t.to_dict() for t in trial_centric_triples],
                'intervention_centric_triples': intervention_output,
                'dynamic_ctkg_triples': dynamic_ctkg_triples,
                'total_trial_centric': len(trial_centric_triples),
                'total_intervention_centric': len(intervention_output),
                'total_dynamic': len(dynamic_ctkg_triples)
            })
            
        except Exception as e:
            logger.error(f"Task 8 failed: {e}")
            return self._create_error_result([str(e)])

    def _rebucket_by_head_level(
        self,
        nct_id: str,
        trial_triples: List[KGTriple],
        intervention_triples: List[KGTriple],
    ) -> (List[KGTriple], List[KGTriple]):
        """
        Reassign triples based on head ID semantics.

        Rule:
        - Trial-level: head == nct_id
        - Intervention-level: head starts with f"{nct_id}_"
        - Others: intervention-level
        """
        merged = (trial_triples or []) + (intervention_triples or [])
        out_trial: List[KGTriple] = []
        out_intervention: List[KGTriple] = []
        seen = set()

        for t in merged:
            key = (
                t.head,
                t.relation,
                t.tail,
                json.dumps(t.attributes or {}, sort_keys=True, ensure_ascii=False),
                json.dumps(t.provenance or {}, sort_keys=True, ensure_ascii=False),
            )
            if key in seen:
                continue
            seen.add(key)

            if t.head == nct_id:
                out_trial.append(t)
            else:
                out_intervention.append(t)

        return out_trial, out_intervention

    def _ensure_provenance_nct_id(self, triples: List[KGTriple], nct_id: str) -> None:
        """Ensure every KGTriple carries provenance.nct_id."""
        for t in triples:
            if not isinstance(t.provenance, dict):
                t.provenance = {}
            t.provenance.setdefault('nct_id', nct_id)

    def _ensure_provenance_nct_id_in_dicts(self, triples: List[Dict], nct_id: str) -> None:
        """Ensure every dict triple carries provenance.nct_id and provenance is last key."""
        for i, t in enumerate(triples):
            if not isinstance(t, dict):
                continue
            prov = t.get('provenance')
            if not isinstance(prov, dict):
                prov = {}
            prov.setdefault('nct_id', nct_id)

            rebuilt = {
                'head': t.get('head', ''),
                'relation': t.get('relation', ''),
                'tail': t.get('tail', ''),
                'head_type': t.get('head_type', ''),
                'tail_type': t.get('tail_type', '')
            }
            if 'tail_attributes' in t and t.get('tail_attributes'):
                rebuilt['tail_attributes'] = t.get('tail_attributes')
            elif 'attributes' in t and t.get('attributes'):
                rebuilt['tail_attributes'] = t.get('attributes')
            rebuilt['provenance'] = prov
            triples[i] = rebuilt
    
    def prepare_input(self, trial_data: Any) -> Dict:
        """Prepare trial data for assembly."""
        return {
            'nct_id': trial_data.nct_id,
            'has_results': trial_data.has_results
        }
    
    # =========================================================================
    # DYNAMIC CTKG INTEGRATION (Task 7)
    # =========================================================================
    
    def _integrate_dynamic_ctkg(
        self,
        trial_centric_ctkg: TrialCentricCTKG,
        task_results: Dict
    ) -> Dict:
        """
        Integrate Task 7 dynamic CTKG data into trial-centric CTKG.
        
        Links version nodes back to canonical NCTID to maintain consistency
        with the trial-centric knowledge graph.
        
        Args:
            trial_centric_ctkg: The trial-centric CTKG being built
            task_results: Results from all tasks including Task 7
            
        Returns:
            Dictionary with integrated version history and triples
        """
        task7 = task_results.get('task7_dynamic_ctkg', {})
        
        if not task7:
            return {'version_history': {}, 'triples': []}
        
        version_nodes = task7.get('version_nodes', [])
        change_relations = task7.get('change_relations', [])
        termination_info = task7.get('termination_info')
        dynamic_triples = task7.get('triples', [])
        
        # Store version history in trial-centric CTKG
        trial_centric_ctkg.version_history = {
            'total_versions': task7.get('total_versions', 0),
            'version_nodes': version_nodes,
            'change_relations': change_relations,
            'termination_info': termination_info,
            'summary': task7.get('summary', {})
        }
        
        # Convert dynamic triples to KGTriple format for consistency
        integrated_triples = []
        for triple_dict in dynamic_triples:
            triple = KGTriple(
                head=triple_dict.get('head', ''),
                relation=triple_dict.get('relation', ''),
                tail=triple_dict.get('tail', ''),
                head_type=triple_dict.get('head_type', ''),
                tail_type=triple_dict.get('tail_type', ''),
                attributes=triple_dict.get('attributes', {}),
                provenance={'source': 'task7_dynamic_ctkg'}
            )
            integrated_triples.append(triple)
            
            # Also add to trial-centric triples for unified view
            trial_centric_ctkg.triples.append(triple)
        
        # Add termination triples if available
        if termination_info:
            nct_id = trial_centric_ctkg.nct_id
            
            # Termination category triple
            if termination_info.get('category'):
                trial_centric_ctkg.triples.append(KGTriple(
                    head=nct_id,
                    relation='hasTerminationCategory',
                    tail=termination_info['category'],
                    head_type='Trial',
                    tail_type='TerminationCategory',
                    attributes={
                        'confidence': termination_info.get('confidence', 1.0),
                        'original_text': termination_info.get('original_text')
                    },
                    provenance={'source': 'task7_dynamic_ctkg'}
                ))
        
        return {
            'version_history': trial_centric_ctkg.version_history,
            'triples': [t.to_dict() for t in integrated_triples]
        }
    
    # =========================================================================
    # TRIAL-CENTRIC CTKG (STRUCTURED)
    # =========================================================================
    
    def _build_trial_centric_structured(
        self, 
        trial_data: Any, 
        task_results: Dict
    ) -> TrialCentricCTKG:
        """
        Build trial-centric CTKG with structured data following Trial_level_information_3.py.
        
        Integrates structured outputs from Task 1-6.
        """
        nct_id = trial_data.nct_id
        
        # 1. Build base information (following Trial_level_information_3.py)
        base_info = self._extract_base_information(trial_data)
        
        # 2. Get conditions
        conditions = trial_data.conditions.get('conditions', [])
        
        # 3. Get structured interventions from Task 2
        interventions = self._get_structured_interventions(trial_data, task_results)
        
        # 4. Get structured eligibility from Task 3
        eligibility = self._get_structured_eligibility(trial_data, task_results)
        
        # 5. Get structured outcomes from Task 1
        outcomes = self._get_structured_outcomes(trial_data, task_results)
        
        # 6. Get inferred purpose from Task 4
        purpose = self._get_inferred_purpose(task_results)
        
        # 7. Get statistical conclusions from Task 5
        conclusions = self._get_statistical_conclusions(task_results)
        
        # 8. Extract baseline characteristics
        baseline = self._extract_baseline_characteristics(trial_data)
        
        # 9. Extract adverse events
        adverse_events = self._extract_adverse_events(trial_data)
        
        # 10. Build triples
        triples = self._build_trial_centric_triples(
            nct_id, trial_data, task_results,
            interventions, outcomes, eligibility, purpose, conclusions, baseline
        )
        
        return TrialCentricCTKG(
            nct_id=nct_id,
            trials_base_information=base_info,
            brief_summary=trial_data.description.get('briefSummary', ''),
            conditions=conditions,
            interventions=interventions,
            eligibility_criteria=eligibility,
            outcomes=outcomes,
            purpose=purpose,
            statistical_conclusions=conclusions,
            group_baseline=baseline,
            adverse_events=adverse_events,
            triples=triples
        )
    
    def _extract_base_information(self, trial_data: Any) -> Dict:
        """
        Extract base trial information following Trial_level_information_3.py schema.
        """
        identification = getattr(trial_data, 'identification', {})
        status = getattr(trial_data, 'status', {})
        design = getattr(trial_data, 'design', {})
        sponsor = getattr(trial_data, 'sponsor_collaborators', {})
        references = getattr(trial_data, 'references', {})
        
        design_info = design.get('designInfo', {})
        
        return {
            'NCT': trial_data.nct_id,
            'officialTitle': identification.get('officialTitle'),
            'briefTitle': identification.get('briefTitle'),
            'overallStatus': status.get('overallStatus'),
            'statusVerifiedDate': status.get('statusVerifiedDate'),
            'startDate': status.get('startDateStruct', {}).get('date'),
            'completionDate': status.get('completionDateStruct', {}).get('date'),
            'leadSponsor': sponsor.get('leadSponsor', {}),
            'collaborators': sponsor.get('collaborators', []),
            'studyType': design.get('studyType'),
            'phases': design.get('phases', []),
            'allocation': design_info.get('allocation', 'N/A'),
            'enrollmentNUM': design.get('enrollmentInfo', {}).get('count', 'N/A'),
            'interventionModel': design_info.get('interventionModel', 'N/A'),
            'primaryPurpose': design_info.get('primaryPurpose', 'N/A'),
            'masking': design_info.get('maskingInfo', {}),
            'references': references.get('references', [])
        }
    
    def _get_structured_interventions(self, trial_data: Any, task_results: Dict) -> Dict:
        """
        Get structured interventions from Task 2 results or raw data.
        
        When Task 2 results are available, uses the LLM-extracted structured format.
        Otherwise, converts raw arm groups and interventions into a consistent structure.
        """
        task2 = task_results.get('task2_intervention_profiling', {})
        profiled = task2.get('profiled_interventions', [])
        
        if profiled:
            # Use Task 2 structured output
            interventions_by_group = {}
            for arm in profiled:
                group_id = arm.get('group_id', '')
                interventions_by_group[group_id] = {
                    'title': arm.get('title', ''),
                    'type': arm.get('type', ''),
                    'description': arm.get('description', ''),
                    'interventions': arm.get('interventions', []),
                    'administration_sequence': arm.get('administration_sequence', 'no order')
                }
            return interventions_by_group
        else:
            # Fall back to raw data - convert to consistent structure
            arms_module = getattr(trial_data, 'arms_interventions', {})
            arm_groups = arms_module.get('armGroups', [])
            raw_interventions = arms_module.get('interventions', [])
            
            # Build intervention lookup by arm label
            intervention_map = {}
            for inv in raw_interventions:
                for arm_label in inv.get('armGroupLabels', []):
                    if arm_label not in intervention_map:
                        intervention_map[arm_label] = []
                    intervention_map[arm_label].append({
                        'name': inv.get('name', ''),
                        'original_text': inv.get('name', ''),
                        'type': inv.get('type', '').lower() if inv.get('type') else 'NA',
                        'dosage_form': 'NA',
                        'administration_route': 'NA',
                        'dosage': 'NA',
                        'frequency': 'NA',
                        'Treatment duration': 'NA',
                        'description': inv.get('description', '')
                    })
            
            # Build structured output
            interventions_by_group = {}
            for i, arm in enumerate(arm_groups):
                group_id = f'AG{i:03d}'
                arm_label = arm.get('label', '')
                interventions_by_group[group_id] = {
                    'title': arm_label,
                    'type': arm.get('type', ''),
                    'description': arm.get('description', ''),
                    'interventions': intervention_map.get(arm_label, []),
                    'administration_sequence': 'no order'
                }
            
            return interventions_by_group
    
    def _get_structured_eligibility(self, trial_data: Any, task_results: Dict) -> Dict:
        """
        Get structured eligibility from Task 3 results or raw data.
        """
        task3 = task_results.get('task3_eligibility_structuring', {})
        structured = task3.get('structured_eligibility', {})
        
        if structured:
            return structured
        else:
            # Fall back to raw eligibility module
            eligibility = getattr(trial_data, 'eligibility', {})
            return {
                'raw_criteria': eligibility.get('eligibilityCriteria', ''),
                'sex': eligibility.get('sex'),
                'minimumAge': eligibility.get('minimumAge'),
                'maximumAge': eligibility.get('maximumAge'),
                'healthyVolunteers': eligibility.get('healthyVolunteers')
            }
    
    def _get_structured_outcomes(self, trial_data: Any, task_results: Dict) -> Dict:
        """
        Get structured outcomes from Task 1 results, merged with raw outcome data.
        
        Following Trial_level_information_3.py outcome_dict structure:
        { outcome_type: { outcome_title: { attributes } } }
        
        If Task 1 results are available, includes:
        - standardized: List of standardized outcome representations
        - core_measurements: Extracted core measurement concepts
        
        Always includes raw outcome attributes from trial data.
        """
        task1 = task_results.get('task1_outcome_standardization', {})
        standardized_list = task1.get('standardized_outcomes', [])

        # Build lookup for Task 1 results by original title
        standardized_lookup = {}
        for outcome in standardized_list:
            title = outcome.get('original_title', '')
            if not title:
                continue

            # Legacy Task1 format (one record per title with nested standardized list)
            if isinstance(outcome.get('standardized'), list):
                standardized_lookup[title] = {
                    'standardized': outcome.get('standardized', []),
                    'core_measurements': [
                        s.get('core_measurement') for s in outcome.get('standardized', [])
                        if isinstance(s, dict) and s.get('core_measurement')
                    ]
                }
                continue

            # New Task1 format (one record per core measurement with attributes)
            if title not in standardized_lookup:
                standardized_lookup[title] = {'standardized': [], 'core_measurements': []}

            core = outcome.get('core_measurement')
            attrs = outcome.get('attributes', {}) or {}
            standardized_lookup[title]['standardized'].append({
                'core_measurement': core,
                'measurement_tool': attrs.get('measurement_tool'),
                'value_condition': attrs.get('value_condition'),
                'conditional_population': attrs.get('conditional_population')
            })
            if core:
                standardized_lookup[title]['core_measurements'].append(core)
        
        outcome_dict = {}
        
        # Always process raw outcome data and merge with Task 1 results
        results = getattr(trial_data, 'results_section', {}) or {}
        om = results.get('outcomeMeasuresModule', {})
        
        for outcome in om.get('outcomeMeasures', []):
            outcome_type = outcome.get('type', 'OTHER')
            title = outcome.get('title', '')
            
            if outcome_type not in outcome_dict:
                outcome_dict[outcome_type] = {}
            
            # Base info from raw data
            outcome_entry = {
                'description': outcome.get('description'),
                'paramType': outcome.get('paramType'),
                'unitOfMeasure': outcome.get('unitOfMeasure'),
                'timeFrame': outcome.get('timeFrame'),
                'is_standardized': title in standardized_lookup
            }
            
            # Merge Task 1 standardization results if available
            if title in standardized_lookup:
                outcome_entry['standardized'] = standardized_lookup[title]['standardized']
                outcome_entry['core_measurements'] = standardized_lookup[title]['core_measurements']
            
            outcome_dict[outcome_type][title] = outcome_entry
        
        return outcome_dict
    
    def _get_inferred_purpose(self, task_results: Dict) -> Dict:
        """Get inferred purpose from Task 4 results."""
        task4 = task_results.get('task4_purpose_inference', {})
        return task4.get('inferred_purpose', {})
    
    def _get_statistical_conclusions(self, task_results: Dict) -> List[Dict]:
        """Get statistical conclusions from Task 5 results."""
        task5 = task_results.get('task5_statistical_conclusions', {})
        return task5.get('conclusions', [])
    
    def _extract_baseline_characteristics(self, trial_data: Any) -> Dict:
        """
        Extract baseline characteristics following Trial_level_information_3.py.
        """
        group_baseline = {}
        
        results = getattr(trial_data, 'results_section', {}) or {}
        baseline = results.get('baselineCharacteristicsModule', {})
        
        if not baseline:
            return group_baseline
        
        groups = baseline.get('groups', [])
        group_info = {g.get('id'): g for g in groups}
        
        measures = baseline.get('measures', [])
        for measure in measures:
            measure_info = {
                'baseline': measure.get('title', ''),
                'parameter': measure.get('paramType'),
                'dispersionType': measure.get('dispersionType'),
                'unitOfMeasure': measure.get('unitOfMeasure')
            }
            
            classes = measure.get('classes', [])
            for cls in classes:
                categories = cls.get('categories', [])
                for category in categories:
                    cat_title = category.get('title', '')
                    measurements = category.get('measurements', [])
                    
                    for measurement in measurements:
                        group_id = measurement.get('groupId')
                        if group_id:
                            if group_id not in group_baseline:
                                group_baseline[group_id] = []
                            
                            entry = measurement.copy()
                            entry.update(measure_info)
                            if cat_title:
                                entry['baseline'] = f"{measure_info['baseline']}, {cat_title}"
                            entry.pop('groupId', None)
                            group_baseline[group_id].append(entry)
        
        return group_baseline
    
    def _extract_adverse_events(self, trial_data: Any) -> Dict:
        """Extract adverse events from results section."""
        results = getattr(trial_data, 'results_section', {}) or {}
        ae_module = results.get('adverseEventsModule', {})
        
        return {
            'frequencyThreshold': ae_module.get('frequencyThreshold'),
            'timeFrame': ae_module.get('timeFrame'),
            'description': ae_module.get('description'),
            'seriousEvents': ae_module.get('seriousEvents', []),
            'otherEvents': ae_module.get('otherEvents', [])
        }
    
    def _build_trial_centric_triples(
        self, 
        nct_id: str, 
        trial_data: Any, 
        task_results: Dict,
        interventions: Dict,
        outcomes: Dict,
        eligibility: Dict,
        purpose: Dict,
        conclusions: List[Dict],
        baseline: Dict = None
    ) -> List[KGTriple]:
        """Build all triples for trial-centric CTKG."""
        triples = []
        
        # Basic design triples
        triples.extend(self._add_design_triples(nct_id, trial_data))
        
        # Condition triples
        triples.extend(self._add_condition_triples(nct_id, trial_data))
        
        # Intervention triples (from structured data)
        triples.extend(self._add_intervention_triples_from_structured(nct_id, interventions))
        
        # Outcome triples (from structured data)
        triples.extend(self._add_outcome_triples_from_structured(nct_id, outcomes, task_results))
        
        # Merge Task3/4 kg triples from upstream task outputs.
        triples.extend(self._add_task3456_kg_triples(task_results))
        
        # Baseline characteristics triples
        if baseline:
            triples.extend(self._add_baseline_triples(nct_id, baseline))
        
        # Adverse event triples
        triples.extend(self._add_adverse_event_triples(nct_id, trial_data))
        
        # Sponsor triples
        triples.extend(self._add_sponsor_triples(nct_id, trial_data))
        
        return triples
    
    def _add_baseline_triples(self, nct_id: str, baseline: Dict) -> List[KGTriple]:
        """
        Add baseline characteristics triples.
        
        Format: (Group_000, hasFemale, 25, {param_type: COUNT_OF_PARTICIPANTS})
        
        Filters out:
        - Empty values
        - Zero values (value == "0" or value == 0)
        - Unknown or Not Reported race categories
        
        Normalizes:
        - Sex: Female, Male fields → hasFemale, hasMale
        
        Args:
            nct_id: NCT ID
            baseline: Baseline characteristics by group
            
        Returns:
            List of KGTriple
        """
        triples = []
        
        for group_id, characteristics in baseline.items():
            # Normalize group ID: BG000 -> NCTID_000
            normalized_group = self._normalize_group_id(group_id, nct_id=nct_id)
            
            for char in characteristics:
                baseline_name = char.get('baseline', '')
                value = char.get('value')
                
                # Skip if no baseline name or value
                if not baseline_name or value is None:
                    continue
                
                # Skip zero values - no meaningful relationship
                if str(value).strip() == '0' or value == 0:
                    continue
                
                # Skip empty string values
                if isinstance(value, str) and not value.strip():
                    continue
                
                # Skip "Unknown or Not Reported" race categories
                if 'Unknown' in baseline_name and 'Not_Reported' in baseline_name.replace(' ', '_'):
                    continue
                if 'Unknown or Not Reported' in baseline_name:
                    continue
                
                # Normalize relation name
                relation = self._normalize_baseline_relation(baseline_name)
                
                # Skip if relation is None (filtered out)
                if relation is None:
                    continue
                
                # Build attributes
                attributes = {}
                if char.get('parameter'):
                    attributes['param_type'] = char.get('parameter')
                if char.get('unitOfMeasure'):
                    attributes['unit'] = char.get('unitOfMeasure')
                if char.get('lowerLimit'):
                    attributes['lower_limit'] = char.get('lowerLimit')
                if char.get('upperLimit'):
                    attributes['upper_limit'] = char.get('upperLimit')
                if char.get('dispersionType'):
                    attributes['dispersion_type'] = char.get('dispersionType')
                if char.get('spread'):
                    attributes['spread'] = char.get('spread')
                
                triples.append(KGTriple(
                    head=normalized_group,
                    relation=relation,
                    tail=str(value),
                    head_type="Group",
                    tail_type="Value",
                    attributes=attributes,
                    provenance={'nct_id': nct_id}
                ))
        
        return triples
    
    def _normalize_baseline_relation(self, baseline_name: str) -> Optional[str]:
        """
        Normalize baseline characteristic name to relation name.
        
        Special handling for:
        - Sex: Female, Male → hasFemale, hasMale
        - Race: specific race → hasRace_{race}
        
        Returns None if the characteristic should be filtered out.
        """
        # Handle Sex fields
        if 'Sex:' in baseline_name or 'Sex/' in baseline_name:
            # "Sex: Female, Male_Female" -> "hasFemale"
            # "Sex: Female, Male_Male" -> "hasMale"
            if 'Female' in baseline_name and baseline_name.endswith('Female'):
                return 'hasFemale'
            elif 'Male' in baseline_name and baseline_name.endswith('Male'):
                return 'hasMale'
            elif baseline_name.endswith('Female'):
                return 'hasFemale'
            elif baseline_name.endswith('Male'):
                return 'hasMale'
        
        # Handle Race fields - skip Unknown/Not Reported
        if 'Race' in baseline_name:
            if 'Unknown' in baseline_name or 'Not Reported' in baseline_name:
                return None
            # Simplify race relation names
            # "Race (NIH/OMB)_Asian" -> "hasRace_Asian"
            race_match = baseline_name.split('_')[-1] if '_' in baseline_name else None
            if race_match:
                return f'hasRace_{race_match}'
        
        # Default: "Age, Continuous" -> "hasAge_Continuous"
        relation = f"has{baseline_name.replace(' ', '_').replace(',', '').replace(':', '')}"
        return relation
    
    def _add_intervention_triples_from_structured(
        self, 
        nct_id: str, 
        interventions: Dict
    ) -> List[KGTriple]:
        """Add intervention triples from structured Task 2 output."""
        triples = []
        
        # Check if this is Task 2 structured format
        for group_id, group_data in interventions.items():
            if isinstance(group_data, dict) and 'interventions' in group_data:
                for intervention in group_data.get('interventions', []):
                    name = intervention.get('name', '')
                    
                    # Skip placebo interventions
                    if self._is_placebo(name):
                        continue
                    
                    if name:
                        intervention_type = intervention.get('type')
                        triples.append(KGTriple(
                            head=nct_id,
                            relation="hasIntervention",
                            tail=name,
                            head_type="Trial",
                            tail_type=self._map_intervention_tail_type(intervention_type)
                        ))
        
        # Fall back format
        if 'armGroups' in interventions:
            for arm in interventions.get('armGroups', []):
                label = arm.get('label', '')
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasArmGroup",
                    tail=label,
                    head_type="Trial",
                    tail_type="ArmGroup",
                    attributes={'type': arm.get('type')}
                ))
        
        return triples
    
    def _add_outcome_triples_from_structured(
        self,
        nct_id: str,
        outcomes: Dict,
        task_results: Optional[Dict] = None
    ) -> List[KGTriple]:
        """Add outcome triples from structured Task 1 output."""
        triples = []

        # Prefer Task 1 standardized output directly:
        # tail = standardized core_measurement
        # tail_attributes = remaining Task 1 fields
        task1 = (task_results or {}).get('task1_outcome_standardization', {})
        standardized_outcomes = task1.get('standardized_outcomes', [])
        if standardized_outcomes:
            for item in standardized_outcomes:
                core = item.get('core_measurement')
                if not core:
                    continue

                attrs = dict(item.get('attributes') or {})
                if item.get('original_title'):
                    attrs['original_title'] = item.get('original_title')

                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasOutcome",
                    tail=core,
                    head_type="Trial",
                    tail_type="Outcome",
                    attributes=attrs
                ))
            return triples
        
        for outcome_type, type_outcomes in outcomes.items():
            for title, outcome_data in type_outcomes.items():
                core_measurements = outcome_data.get('core_measurements', []) or []
                standardized_entries = outcome_data.get('standardized', []) or []

                # If standardized cores exist, emit one hasOutcome per core measurement
                if core_measurements:
                    for idx, core in enumerate(core_measurements):
                        if not core:
                            continue
                        std_entry = standardized_entries[idx] if idx < len(standardized_entries) else {}
                        attrs = {
                            'original_title': title,
                            'outcome_type': outcome_type,
                            'measurement_tool': std_entry.get('measurement_tool') if isinstance(std_entry, dict) else None,
                            'value_condition': std_entry.get('value_condition') if isinstance(std_entry, dict) else None,
                            'conditional_population': std_entry.get('conditional_population') if isinstance(std_entry, dict) else None,
                            'paramType': outcome_data.get('paramType'),
                            'timeFrame': outcome_data.get('timeFrame'),
                            'unitOfMeasure': outcome_data.get('unitOfMeasure'),
                            'units': outcome_data.get('unitOfMeasure'),
                            'description': outcome_data.get('description')
                        }
                        triples.append(KGTriple(
                            head=nct_id,
                            relation="hasOutcome",
                            tail=core,
                            head_type="Trial",
                            tail_type="Outcome",
                            attributes=attrs
                        ))
                    continue

                # Last-resort fallback when no standardized core exists
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasOutcome",
                    tail=title,
                    head_type="Trial",
                    tail_type="Outcome",
                    attributes={
                        'original_title': title,
                        'outcome_type': outcome_type,
                        'measurement_tool': None,
                        'value_condition': None,
                        'conditional_population': None,
                        'paramType': outcome_data.get('paramType'),
                        'timeFrame': outcome_data.get('timeFrame'),
                        'unitOfMeasure': outcome_data.get('unitOfMeasure'),
                        'units': outcome_data.get('unitOfMeasure'),
                        'description': outcome_data.get('description')
                    }
                ))
        
        return triples
    
    def _add_purpose_triples(self, nct_id: str, purpose: Dict) -> List[KGTriple]:
        """Add purpose triples from Task 4 output."""
        triples = []
        
        if purpose:
            # Primary purpose
            primary = purpose.get('primary_purpose', '')
            if primary:
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasPurpose",
                    tail=primary,
                    head_type="Trial",
                    tail_type="Purpose"
                ))
            
            # Clinical goals
            for goal in purpose.get('clinical_goals', []):
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasClinicalGoal",
                    tail=goal,
                    head_type="Trial",
                    tail_type="ClinicalGoal"
                ))
        
        return triples
    
    def _add_conclusion_triples(self, nct_id: str, conclusions: List[Dict]) -> List[KGTriple]:
        """Add statistical conclusion triples from Task 5 output."""
        triples = []
        
        for conclusion in conclusions:
            triplet = conclusion.get('triplet', {})
            is_significant = conclusion.get('is_statistically_significant', False)
            
            if triplet and is_significant:
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasStatisticalConclusion",
                    tail=triplet.get('tail_entity_original', ''),
                    head_type="Trial",
                    tail_type="StatisticalConclusion",
                    attributes={
                        'intervention': triplet.get('head_entity_name'),
                        'comparator': triplet.get('head_entity_2_name'),
                        'relation': triplet.get('comparative_relation'),
                        'context': triplet.get('relationship_context'),
                        'p_value': conclusion.get('p_value')
                    }
                ))
        
        return triples

    def _add_task3456_kg_triples(self, task_results: Dict) -> List[KGTriple]:
        """
        Merge Task3/4 kg_triples into Task8 trial-centric triples.

        Task5/6 triples are merged into intervention_centric_triples.
        """
        triples: List[KGTriple] = []
        seen = set()

        def _append_hrt(src_triple: Dict, source: str) -> None:
            if not isinstance(src_triple, dict):
                return
            head = src_triple.get("head")
            rel = src_triple.get("relation")
            tail = src_triple.get("tail")
            if not head or not rel or tail is None or tail == "":
                return
            key = (str(head), str(rel), str(tail), source)
            if key in seen:
                return
            seen.add(key)

            attrs = src_triple.get("tail_attributes")
            if attrs is None:
                attrs = src_triple.get("attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            if source:
                attrs = dict(attrs)
                attrs.setdefault("source_task", source)

            triples.append(KGTriple(
                head=str(head),
                relation=str(rel),
                tail=str(tail),
                head_type=src_triple.get("head_type", ""),
                tail_type=src_triple.get("tail_type", ""),
                attributes=attrs,
                provenance=src_triple.get("provenance", {}) or {"source": source}
            ))

        # Task3: structured_eligibility.kg_triples (HRT)
        task3 = task_results.get("task3_eligibility_structuring", {}) or {}
        task3_kg = ((task3.get("structured_eligibility") or {}).get("kg_triples") or [])
        for t in task3_kg:
            _append_hrt(t, "task3_eligibility_structuring")

        # Task4: inferred_purpose.kg_triples (HRT)
        task4 = task_results.get("task4_purpose_inference", {}) or {}
        task4_kg = ((task4.get("inferred_purpose") or {}).get("kg_triples") or [])
        for t in task4_kg:
            _append_hrt(t, "task4_purpose_inference")

        return triples

    def _get_task56_raw_triples(self, task_results: Dict) -> List[Dict]:
        """Return Task5/Task6 kg_triples in original raw schema."""
        raw: List[Dict] = []
        task5 = task_results.get("task5_statistical_conclusions", {}) or {}
        raw.extend([t for t in (task5.get("kg_triples") or []) if isinstance(t, dict)])
        task6 = task_results.get("task6_disease_mapping", {}) or {}
        raw.extend([t for t in (task6.get("kg_triples") or []) if isinstance(t, dict)])
        return raw

    def _get_task2_raw_relations(self, task_results: Dict) -> List[Dict]:
        """Return Task2 co_treat/composite relations in original raw schema."""
        raw: List[Dict] = []
        task2 = task_results.get("task2_intervention_profiling", {}) or {}
        raw.extend([t for t in (task2.get("co_treat_relations") or []) if isinstance(t, dict)])
        raw.extend([t for t in (task2.get("composite_relations") or []) if isinstance(t, dict)])
        return raw
    
    def _add_sponsor_triples(self, nct_id: str, trial_data: Any) -> List[KGTriple]:
        """Add sponsor and collaborator triples."""
        triples = []
        
        sponsor = getattr(trial_data, 'sponsor_collaborators', {})
        
        lead = sponsor.get('leadSponsor', {})
        if lead.get('name'):
            triples.append(KGTriple(
                head=nct_id,
                relation="hasLeadSponsor",
                tail=lead.get('name'),
                head_type="Trial",
                tail_type="Organization",
                attributes={'class': lead.get('class')}
            ))
        
        for collab in sponsor.get('collaborators', []):
            if collab.get('name'):
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasCollaborator",
                    tail=collab.get('name'),
                    head_type="Trial",
                    tail_type="Organization",
                    attributes={'class': collab.get('class')}
                ))
        
        return triples
    
    def _add_design_triples(self, nct_id: str, trial_data: Any) -> List[KGTriple]:
        """Add design-related triples."""
        triples = []
        design = getattr(trial_data, 'design', {})
        status = getattr(trial_data, 'status', {})
        
        # Phase
        phases = design.get('phases', [])
        for phase in phases if isinstance(phases, list) else [phases]:
            if phase:
                triples.append(KGTriple(
                    head=nct_id,
                    relation="hasPhase",
                    tail=phase,
                    head_type="Trial",
                    tail_type="Phase"
                ))
        
        # Study type
        study_type = design.get('studyType')
        if study_type:
            triples.append(KGTriple(
                head=nct_id,
                relation="hasStudyType",
                tail=study_type,
                head_type="Trial",
                tail_type="StudyType"
            ))
        
        # Status
        overall_status = status.get('overallStatus')
        if overall_status:
            triples.append(KGTriple(
                head=nct_id,
                relation="hasStatus",
                tail=overall_status,
                head_type="Trial",
                tail_type="Status"
            ))
        
        return triples
    
    def _add_condition_triples(self, nct_id: str, trial_data: Any) -> List[KGTriple]:
        """Add condition-related triples."""
        triples = []
        conditions = getattr(trial_data, 'conditions', {})
        
        for condition in conditions.get('conditions', []):
            triples.append(KGTriple(
                head=nct_id,
                relation="hasCondition",
                tail=condition,
                head_type="Trial",
                tail_type="Condition"
            ))
        
        return triples
    
    def _add_eligibility_triples(
        self, 
        nct_id: str, 
        trial_data: Any,
        task_results: Dict
    ) -> List[KGTriple]:
        """Add eligibility-related triples."""
        triples = []
        
        eligibility = getattr(trial_data, 'eligibility', {})
        
        # Age constraints
        min_age = eligibility.get('minimumAge')
        max_age = eligibility.get('maximumAge')
        
        if min_age:
            triples.append(KGTriple(
                head=nct_id,
                relation="hasMinimumAge",
                tail=min_age,
                head_type="Trial",
                tail_type="AgeConstraint"
            ))
        
        if max_age:
            triples.append(KGTriple(
                head=nct_id,
                relation="hasMaximumAge",
                tail=max_age,
                head_type="Trial",
                tail_type="AgeConstraint"
            ))
        
        # Sex constraint
        sex = eligibility.get('sex')
        if sex and sex != 'ALL':
            triples.append(KGTriple(
                head=nct_id,
                relation="requiresSex",
                tail=sex,
                head_type="Trial",
                tail_type="SexConstraint"
            ))
        
        return triples
    
    def _add_adverse_event_triples(
        self, 
        nct_id: str, 
        trial_data: Any
    ) -> List[KGTriple]:
        """
        Add adverse event-related triples.
        
        Creates:
        1. Trial-level AE triples with aggregated statistics
        2. Group-level AE triples for each group
        """
        triples = []
        
        ae_module = getattr(trial_data, 'adverse_events', {})
        
        # Process serious adverse events
        serious_events = ae_module.get('seriousEvents', [])
        triples.extend(self._process_ae_events(
            nct_id, serious_events, 'hasSeriousAdverseEvent'
        ))
        
        # Process other adverse events
        other_events = ae_module.get('otherEvents', [])
        triples.extend(self._process_ae_events(
            nct_id, other_events, 'hasAdverseEvent'
        ))
        
        return triples
    
    def _process_ae_events(
        self,
        nct_id: str,
        events: List[Dict],
        relation: str
    ) -> List[KGTriple]:
        """
        Process adverse events and create triples.
        
        Creates:
        1. Trial-level triple with total statistics
        2. Group-level triples for each group with non-zero affected
        """
        triples = []
        
        for event in events:
            term = event.get('term', '')
            if not term:
                continue
            
            organ_system = event.get('organSystem', '')
            stats = event.get('stats', [])
            
            # Calculate trial-level statistics (sum across all groups)
            total_affected = 0
            total_at_risk = 0
            
            for stat in stats:
                num_affected = stat.get('numAffected', 0) or 0
                num_at_risk = stat.get('numAtRisk', 0) or 0
                total_affected += num_affected
                total_at_risk += num_at_risk
            
            # Only create trial-level triple if someone was affected
            if total_affected > 0:
                rate = round(total_affected / total_at_risk * 100, 2) if total_at_risk > 0 else 0
                
                trial_attrs = {
                    'total_affected': total_affected,
                    'total_at_risk': total_at_risk,
                    'rate_percent': rate,
                }
                if organ_system:
                    trial_attrs['organ_system'] = organ_system

                triples.append(KGTriple(
                    head=nct_id,
                    relation=relation,
                    tail=term,
                    head_type="Trial",
                    tail_type="Disease",
                    attributes=trial_attrs
                ))
            
            # Create group-level triples (only for groups with affected > 0)
            for stat in stats:
                group_id = stat.get('groupId', '')
                num_affected = stat.get('numAffected', 0) or 0
                num_at_risk = stat.get('numAtRisk', 0) or 0
                
                # Skip if no one affected in this group
                if num_affected == 0:
                    continue
                
                # Normalize group ID: EG000 -> NCTID_000
                normalized_group = self._normalize_group_id(group_id, nct_id=nct_id)
                rate = round(num_affected / num_at_risk * 100, 2) if num_at_risk > 0 else 0
                
                group_attrs = {
                    'num_affected': num_affected,
                    'num_at_risk': num_at_risk,
                    'rate_percent': rate,
                }
                if organ_system:
                    group_attrs['organ_system'] = organ_system

                triples.append(KGTriple(
                    head=normalized_group,
                    relation=relation,
                    tail=term,
                    head_type="Group",
                    tail_type="Disease",
                    attributes=group_attrs,
                    provenance={'nct_id': nct_id}
                ))
        
        return triples
    
    def _normalize_group_id(self, group_id: str, nct_id: Optional[str] = None) -> str:
        """
        Normalize group ID to unified format.
        
        BG000, OG000, EG000, AG000 -> <NCTID>_000 (when nct_id is provided)
        
        Args:
            group_id: Original group ID
            nct_id: Trial NCT ID for normalized prefix
            
        Returns:
            Normalized group ID in format <NCTID>_XXX (preferred)
        """
        if not group_id:
            return group_id

        # Keep already-normalized NCT group IDs unchanged.
        if isinstance(group_id, str) and group_id.startswith("NCT") and "_" in group_id:
            return group_id
        
        # Extract the numeric part
        # Handles: BG000, OG000, EG000, AG000, etc.
        import re
        match = re.match(r'[A-Z]{1,2}(\d+)', group_id)
        if match:
            num = match.group(1)
            if nct_id:
                return f"{nct_id}_{num}"
            return f"Group_{num}"
        
        return group_id
    
    def _is_placebo(self, intervention_name: str) -> bool:
        """
        Check if an intervention is a placebo.
        
        Args:
            intervention_name: Name of the intervention
            
        Returns:
            True if the intervention is a placebo
        """
        if not intervention_name:
            return False
        
        name_lower = intervention_name.lower().strip()
        
        # Direct match
        if name_lower == 'placebo':
            return True
        
        # Check for placebo variants
        placebo_patterns = [
            'placebo',
            'matching placebo',
            'placebo control',
            'sham',
            'dummy',
        ]
        
        for pattern in placebo_patterns:
            if pattern in name_lower:
                return True
        
        return False

    def _map_intervention_tail_type(self, intervention_type: Optional[str]) -> str:
        """Map intervention type to specific biomedical entity type."""
        if not intervention_type:
            return "Intervention"

        t = str(intervention_type).strip().lower()
        mapping = {
            'drug': 'Drug',
            'biological': 'Biological',
            'device': 'Device',
            'procedure': 'Procedure',
            'dietary supplement': 'DietarySupplement',
            'dietary_supplement': 'DietarySupplement',
            'behavioral': 'BehavioralIntervention',
            'genetic': 'GeneticIntervention',
            'radiation': 'RadiationTherapy',
            'diagnostic test': 'DiagnosticTest',
            'diagnostic_test': 'DiagnosticTest',
            'combination product': 'CombinationProduct',
            'combination_product': 'CombinationProduct',
            'other': 'Intervention'
        }
        return mapping.get(t, "Intervention")
    
    # =========================================================================
    # INTERVENTION-CENTRIC CTKG
    # =========================================================================
    
    def _build_intervention_centric(
        self, 
        trial_data: Any,
        task_results: Dict
    ) -> List[KGTriple]:
        """
        Build intervention-centric CTKG centered on interventions.
        
        Creates triples for:
        - Intervention-outcome relations
        - Intervention-disease relations
        - Co-treatment relations
        - Adverse event relations
        """
        triples = []
        nct_id = trial_data.nct_id
        
        # Get profiled interventions
        task2 = task_results.get('task2_intervention_profiling', {})
        profiled = task2.get('profiled_interventions', [])
        
        # Get statistical conclusions
        task5 = task_results.get('task5_statistical_conclusions', {})
        conclusions = task5.get('conclusions', [])
        
        # Get disease mappings
        task6 = task_results.get('task6_disease_mapping', {})
        disease_relations = task6.get('disease_relations', [])
        
        # Intervention-outcome relations from conclusions
        for conclusion in conclusions:
            triplet = conclusion.get('triplet', {})
            if conclusion.get('statistical_significance'):
                intervention = triplet.get('head_entity_name', '')
                outcome = triplet.get('tail_entity_original', '')
                relation_context = triplet.get('relationship_context', '')
                
                # Determine relation type
                if 'decreas' in relation_context.lower():
                    relation = 'decreases'
                elif 'increas' in relation_context.lower():
                    relation = 'increases'
                elif 'improv' in relation_context.lower():
                    relation = 'improves'
                else:
                    relation = 'affects'
                
                triples.append(KGTriple(
                    head=intervention,
                    relation=relation,
                    tail=outcome,
                    head_type="Intervention",
                    tail_type="Outcome",
                    provenance={'nct_id': nct_id, 'p_value': conclusion.get('p_value')}
                ))
        
        # Intervention-disease relations
        for mapping in disease_relations:
            intervention = mapping.get('intervention', '')
            disease = mapping.get('disease', '')
            relation = mapping.get('intervention_disease_relation', 'affects')
            
            triples.append(KGTriple(
                head=intervention,
                relation=relation,
                tail=disease,
                head_type="Intervention",
                tail_type="Disease",
                attributes={
                    'clinical_domain': mapping.get('clinical_domain'),
                    'evidence_strength': mapping.get('evidence_strength')
                },
                provenance={'nct_id': nct_id}
            ))
        
        # Co-treatment relations from Task2 (new schema)
        co_treat = task2.get('co_treat_relations', []) or []
        for rel in co_treat:
            head = rel.get('head')
            tail = rel.get('tail')
            relation = rel.get('relation') or 'co_treat'
            if not head or not tail:
                continue
            triples.append(KGTriple(
                head=head,
                relation=relation,
                tail=tail,
                head_type=rel.get('head_type', 'Drug'),
                tail_type=rel.get('tail_type', 'Drug'),
                attributes={
                    'head_attributes': rel.get('head_attributes') or {},
                    'tail_attributes': rel.get('tail_attributes') or {},
                    'relation_attributes': rel.get('attributes') or {},
                    'source_task': 'task2_intervention_profiling'
                },
                provenance={'nct_id': nct_id}
            ))

        # Composite relations from Task2 (belongs_to_group)
        composite = task2.get('composite_relations', []) or []
        for rel in composite:
            head = rel.get('head')
            tail = rel.get('tail')
            relation = rel.get('relation') or 'belongs_to_group'
            if not head or not tail:
                continue
            triples.append(KGTriple(
                head=head,
                relation=relation,
                tail=tail,
                head_type=rel.get('head_type', 'Drug'),
                tail_type=rel.get('tail_type', 'Group'),
                attributes={
                    'head_attributes': rel.get('head_attributes') or {},
                    'tail_attributes': rel.get('tail_attributes') or {},
                    'relation_attributes': rel.get('attributes') or {},
                    'source_task': 'task2_intervention_profiling'
                },
                provenance={'nct_id': nct_id}
            ))
        
        return triples

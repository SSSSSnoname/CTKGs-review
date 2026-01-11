"""
CTKG Construction Agent

A modular agent (controller: GPT-5) that converts ClinicalTrials.gov records 
into structured Clinical Trial Knowledge Graphs (CTKGs).

The agent can process:
- Trial-centric CTKG: study design, eligibility, outcomes, and adverse events
- Intervention-centric CTKG: interventions to outcomes, diseases, safety events
- Dynamic CTKG: protocol modifications and version histories

Given one or more NCT identifiers, the agent retrieves registry records and 
flexibly executes a subset of nine modular tasks depending on the scope.
"""

import json
import logging
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of available tasks in the CTKG Construction Agent."""
    TASK1_OUTCOME_STANDARDIZATION = "task1_outcome_standardization"
    TASK2_INTERVENTION_PROFILING = "task2_intervention_profiling"
    TASK3_ELIGIBILITY_STRUCTURING = "task3_eligibility_structuring"
    TASK4_PURPOSE_INFERENCE = "task4_purpose_inference"
    TASK5_STATISTICAL_CONCLUSIONS = "task5_statistical_conclusions"
    TASK6_DISEASE_MAPPING = "task6_disease_mapping"
    TASK7_DYNAMIC_CTKG = "task7_dynamic_ctkg"
    TASK8_CTKG_ASSEMBLY = "task8_ctkg_assembly"
    TASK9_ENTITY_LINKING = "task9_entity_linking"


class CTKGType(Enum):
    """Types of CTKGs that can be generated."""
    TRIAL_CENTRIC = "trial_centric"
    INTERVENTION_CENTRIC = "intervention_centric"
    DYNAMIC = "dynamic"


@dataclass
class TrialData:
    """
    Container for parsed clinical trial data.
    
    Following Trial_level_information_3.py schema for comprehensive trial representation.
    """
    nct_id: str
    protocol_section: Dict = field(default_factory=dict)
    results_section: Dict = field(default_factory=dict)
    derived_section: Dict = field(default_factory=dict)
    has_results: bool = False
    
    # Extracted modules from protocol section
    identification: Dict = field(default_factory=dict)
    status: Dict = field(default_factory=dict)
    description: Dict = field(default_factory=dict)
    conditions: Dict = field(default_factory=dict)
    design: Dict = field(default_factory=dict)
    arms_interventions: Dict = field(default_factory=dict)
    eligibility: Dict = field(default_factory=dict)
    outcomes: Dict = field(default_factory=dict)
    sponsor_collaborators: Dict = field(default_factory=dict)
    references: Dict = field(default_factory=dict)
    contacts_locations: Dict = field(default_factory=dict)
    
    # Extracted modules from results section
    adverse_events: Dict = field(default_factory=dict)
    baseline_characteristics: Dict = field(default_factory=dict)
    outcome_measures: Dict = field(default_factory=dict)
    participant_flow: Dict = field(default_factory=dict)
    
    @classmethod
    def from_raw(cls, raw_data: Dict) -> 'TrialData':
        """Create TrialData from raw ClinicalTrials.gov JSON."""
        protocol = raw_data.get('protocolSection', {})
        results = raw_data.get('resultsSection', {})
        
        trial = cls(
            nct_id=protocol.get('identificationModule', {}).get('nctId', ''),
            protocol_section=protocol,
            results_section=results,
            derived_section=raw_data.get('derivedSection', {}),
            has_results=raw_data.get('hasResults', False)
        )
        
        # Parse protocol modules
        trial.identification = protocol.get('identificationModule', {})
        trial.status = protocol.get('statusModule', {})
        trial.description = protocol.get('descriptionModule', {})
        trial.conditions = protocol.get('conditionsModule', {})
        trial.design = protocol.get('designModule', {})
        trial.arms_interventions = protocol.get('armsInterventionsModule', {})
        trial.eligibility = protocol.get('eligibilityModule', {})
        trial.outcomes = protocol.get('outcomesModule', {})
        trial.sponsor_collaborators = protocol.get('sponsorCollaboratorsModule', {})
        trial.references = protocol.get('referencesModule', {})
        trial.contacts_locations = protocol.get('contactsLocationsModule', {})
        
        # Parse results modules (if available)
        if results:
            trial.adverse_events = results.get('adverseEventsModule', {})
            trial.baseline_characteristics = results.get('baselineCharacteristicsModule', {})
            trial.outcome_measures = results.get('outcomeMeasuresModule', {})
            trial.participant_flow = results.get('participantFlowModule', {})
        
        return trial
    
    def get_base_information(self) -> Dict:
        """
        Get base trial information following Trial_level_information_3.py schema.
        
        Returns:
            Dictionary containing key trial metadata
        """
        design_info = self.design.get('designInfo', {})
        
        return {
            'NCT': self.nct_id,
            'officialTitle': self.identification.get('officialTitle'),
            'briefTitle': self.identification.get('briefTitle'),
            'overallStatus': self.status.get('overallStatus'),
            'statusVerifiedDate': self.status.get('statusVerifiedDate'),
            'startDate': self.status.get('startDateStruct', {}).get('date'),
            'completionDate': self.status.get('completionDateStruct', {}).get('date'),
            'leadSponsor': self.sponsor_collaborators.get('leadSponsor', {}),
            'studyType': self.design.get('studyType'),
            'phases': self.design.get('phases', []),
            'allocation': design_info.get('allocation', 'N/A'),
            'enrollmentNUM': self.design.get('enrollmentInfo', {}).get('count', 'N/A'),
            'interventionModel': design_info.get('interventionModel', 'N/A'),
            'primaryPurpose': design_info.get('primaryPurpose', 'N/A'),
            'references': self.references.get('references', [])
        }


@dataclass
class TaskResult:
    """Container for task execution results."""
    task_type: TaskType
    success: bool
    data: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class CTKGTriple:
    """Representation of a knowledge graph triple."""
    head: str
    relation: str
    tail: str
    head_type: str = ""
    tail_type: str = ""
    attributes: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)


class CTKGConstructionAgent:
    """
    Main agent class for constructing Clinical Trial Knowledge Graphs.
    
    This agent orchestrates the execution of 9 modular tasks to convert
    ClinicalTrials.gov records into structured knowledge graphs.
    """
    
    def __init__(
        self,
        llm_client=None,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the CTKG Construction Agent.
        
        Args:
            llm_client: Pre-configured LLM client (optional)
            api_key: API key for LLM service
            model: Model name to use (default: gpt-5)
            cache_dir: Directory for caching intermediate results
        """
        self.llm_client = llm_client
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir or Path("./cache")
        
        # Task handlers (initialized lazily)
        self._task_handlers = {}
        
        # Results storage
        self.trial_data_cache: Dict[str, TrialData] = {}
        self.task_results: Dict[str, Dict[TaskType, TaskResult]] = {}
        
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_trial_from_api(self, nct_id: str) -> TrialData:
        """
        Download trial data from ClinicalTrials.gov API.
        
        Args:
            nct_id: NCT identifier (e.g., "NCT00256997")
            
        Returns:
            TrialData object containing parsed trial information
        """
        from .data_loader.ctgov_api import fetch_trial_by_nct_id
        
        raw_data = fetch_trial_by_nct_id(nct_id)
        trial_data = TrialData.from_raw(raw_data)
        self.trial_data_cache[nct_id] = trial_data
        
        logger.info(f"Loaded trial {nct_id} from ClinicalTrials.gov API")
        return trial_data
    
    def load_trial_from_file(self, file_path: Union[str, Path], nct_id: str) -> TrialData:
        """
        Load trial data from local JSON file.
        
        Args:
            file_path: Path to JSON file containing trial data
            nct_id: NCT identifier to extract from the file
            
        Returns:
            TrialData object containing parsed trial information
        """
        from .data_loader.file_loader import load_trial_from_json
        
        raw_data = load_trial_from_json(file_path, nct_id)
        trial_data = TrialData.from_raw(raw_data)
        self.trial_data_cache[nct_id] = trial_data
        
        logger.info(f"Loaded trial {nct_id} from {file_path}")
        return trial_data
    
    def load_trials_batch(
        self, 
        nct_ids: List[str], 
        source: str = "api",
        file_path: Optional[Path] = None
    ) -> Dict[str, TrialData]:
        """
        Load multiple trials in batch.
        
        Args:
            nct_ids: List of NCT identifiers
            source: Data source ("api" or "file")
            file_path: Path to JSON file (required if source="file")
            
        Returns:
            Dictionary mapping NCT IDs to TrialData objects
        """
        results = {}
        
        for nct_id in nct_ids:
            try:
                if source == "api":
                    results[nct_id] = self.load_trial_from_api(nct_id)
                elif source == "file" and file_path:
                    results[nct_id] = self.load_trial_from_file(file_path, nct_id)
            except Exception as e:
                logger.error(f"Failed to load trial {nct_id}: {e}")
                
        return results
    
    # =========================================================================
    # TASK EXECUTION
    # =========================================================================
    
    def execute_task(
        self, 
        nct_id: str, 
        task_type: TaskType,
        **kwargs
    ) -> TaskResult:
        """
        Execute a single task for a given trial.
        
        Args:
            nct_id: NCT identifier
            task_type: Type of task to execute
            **kwargs: Additional task-specific parameters
            
        Returns:
            TaskResult containing the execution results
        """
        if nct_id not in self.trial_data_cache:
            raise ValueError(f"Trial {nct_id} not loaded. Call load_trial_* first.")
        
        trial_data = self.trial_data_cache[nct_id]
        
        # Get or create task handler
        handler = self._get_task_handler(task_type)
        
        # Execute task
        try:
            result = handler.execute(trial_data, **kwargs)
            
            # Cache result
            if nct_id not in self.task_results:
                self.task_results[nct_id] = {}
            self.task_results[nct_id][task_type] = result
            
            logger.info(f"Completed {task_type.value} for {nct_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute {task_type.value} for {nct_id}: {e}")
            return TaskResult(
                task_type=task_type,
                success=False,
                errors=[str(e)]
            )
    
    def execute_pipeline(
        self,
        nct_id: str,
        tasks: Optional[List[TaskType]] = None,
        ctkg_type: CTKGType = CTKGType.TRIAL_CENTRIC
    ) -> Dict[TaskType, TaskResult]:
        """
        Execute a pipeline of tasks for a given trial.
        
        Args:
            nct_id: NCT identifier
            tasks: List of tasks to execute (None = all applicable tasks)
            ctkg_type: Type of CTKG to construct
            
        Returns:
            Dictionary mapping task types to their results
        """
        if tasks is None:
            tasks = self._get_default_tasks_for_ctkg_type(ctkg_type)
        
        results = {}
        
        for task_type in tasks:
            # Check dependencies
            dependencies = self._get_task_dependencies(task_type)
            for dep in dependencies:
                if dep not in results or not results[dep].success:
                    logger.warning(f"Skipping {task_type.value}: dependency {dep.value} not satisfied")
                    continue
            
            # Execute task
            result = self.execute_task(nct_id, task_type)
            results[task_type] = result
            
        return results
    
    def _get_task_handler(self, task_type: TaskType):
        """Get or create a task handler for the given task type."""
        if task_type not in self._task_handlers:
            self._task_handlers[task_type] = self._create_task_handler(task_type)
        return self._task_handlers[task_type]
    
    def _create_task_handler(self, task_type: TaskType):
        """Create a new task handler instance."""
        from .tasks import (
            Task1OutcomeStandardization,
            Task2InterventionProfiling,
            Task3EligibilityStructuring,
            Task4PurposeInference,
            Task5StatisticalConclusions,
            Task6DiseaseMapping,
            Task7DynamicCTKG,
            Task8CTKGAssembly,
            Task9EntityLinking
        )
        
        handlers = {
            TaskType.TASK1_OUTCOME_STANDARDIZATION: Task1OutcomeStandardization,
            TaskType.TASK2_INTERVENTION_PROFILING: Task2InterventionProfiling,
            TaskType.TASK3_ELIGIBILITY_STRUCTURING: Task3EligibilityStructuring,
            TaskType.TASK4_PURPOSE_INFERENCE: Task4PurposeInference,
            TaskType.TASK5_STATISTICAL_CONCLUSIONS: Task5StatisticalConclusions,
            TaskType.TASK6_DISEASE_MAPPING: Task6DiseaseMapping,
            TaskType.TASK7_DYNAMIC_CTKG: Task7DynamicCTKG,
            TaskType.TASK8_CTKG_ASSEMBLY: Task8CTKGAssembly,
            TaskType.TASK9_ENTITY_LINKING: Task9EntityLinking,
        }
        
        handler_class = handlers.get(task_type)
        if handler_class is None:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return handler_class(
            llm_client=self.llm_client,
            api_key=self.api_key,
            model=self.model
        )
    
    def _get_default_tasks_for_ctkg_type(self, ctkg_type: CTKGType) -> List[TaskType]:
        """Get default task list for a given CTKG type."""
        if ctkg_type == CTKGType.TRIAL_CENTRIC:
            return [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING,
                TaskType.TASK3_ELIGIBILITY_STRUCTURING,
                TaskType.TASK4_PURPOSE_INFERENCE,
                TaskType.TASK5_STATISTICAL_CONCLUSIONS,
                TaskType.TASK8_CTKG_ASSEMBLY,
                TaskType.TASK9_ENTITY_LINKING,
            ]
        elif ctkg_type == CTKGType.INTERVENTION_CENTRIC:
            return [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING,
                TaskType.TASK5_STATISTICAL_CONCLUSIONS,
                TaskType.TASK6_DISEASE_MAPPING,
                TaskType.TASK8_CTKG_ASSEMBLY,
                TaskType.TASK9_ENTITY_LINKING,
            ]
        elif ctkg_type == CTKGType.DYNAMIC:
            return [
                TaskType.TASK7_DYNAMIC_CTKG,
                TaskType.TASK9_ENTITY_LINKING,
            ]
        return []
    
    def _get_task_dependencies(self, task_type: TaskType) -> List[TaskType]:
        """Get task dependencies."""
        dependencies = {
            TaskType.TASK1_OUTCOME_STANDARDIZATION: [],
            TaskType.TASK2_INTERVENTION_PROFILING: [],
            TaskType.TASK3_ELIGIBILITY_STRUCTURING: [],
            TaskType.TASK4_PURPOSE_INFERENCE: [],
            TaskType.TASK5_STATISTICAL_CONCLUSIONS: [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING
            ],
            TaskType.TASK6_DISEASE_MAPPING: [
                TaskType.TASK5_STATISTICAL_CONCLUSIONS
            ],
            TaskType.TASK7_DYNAMIC_CTKG: [],
            TaskType.TASK8_CTKG_ASSEMBLY: [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING
            ],
            TaskType.TASK9_ENTITY_LINKING: [
                TaskType.TASK8_CTKG_ASSEMBLY
            ],
        }
        return dependencies.get(task_type, [])
    
    # =========================================================================
    # CTKG GENERATION
    # =========================================================================
    
    def generate_trial_centric_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """
        Generate trial-centric CTKG for a given trial.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of CTKGTriple objects representing the knowledge graph
        """
        results = self.execute_pipeline(nct_id, ctkg_type=CTKGType.TRIAL_CENTRIC)
        
        if TaskType.TASK8_CTKG_ASSEMBLY in results:
            return results[TaskType.TASK8_CTKG_ASSEMBLY].data.get('triples', [])
        return []
    
    def generate_intervention_centric_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """
        Generate intervention-centric CTKG for a given trial.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of CTKGTriple objects representing the knowledge graph
        """
        results = self.execute_pipeline(nct_id, ctkg_type=CTKGType.INTERVENTION_CENTRIC)
        
        if TaskType.TASK8_CTKG_ASSEMBLY in results:
            return results[TaskType.TASK8_CTKG_ASSEMBLY].data.get('triples', [])
        return []
    
    def generate_dynamic_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """
        Generate dynamic CTKG (version history) for a given trial.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of CTKGTriple objects representing version changes
        """
        results = self.execute_pipeline(nct_id, ctkg_type=CTKGType.DYNAMIC)
        
        if TaskType.TASK7_DYNAMIC_CTKG in results:
            return results[TaskType.TASK7_DYNAMIC_CTKG].data.get('triples', [])
        return []
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_ctkg(
        self,
        triples: List[CTKGTriple],
        output_path: Union[str, Path],
        format: str = "json"
    ):
        """
        Export CTKG to file.
        
        Args:
            triples: List of CTKGTriple objects
            output_path: Output file path
            format: Output format ("json", "jsonl", "csv", "rdf")
        """
        output_path = Path(output_path)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([self._triple_to_dict(t) for t in triples], f, indent=2)
        elif format == "jsonl":
            with open(output_path, 'w') as f:
                for triple in triples:
                    f.write(json.dumps(self._triple_to_dict(triple)) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(triples)} triples to {output_path}")
    
    def _triple_to_dict(self, triple: CTKGTriple) -> Dict:
        """Convert CTKGTriple to dictionary."""
        return {
            "head": triple.head,
            "relation": triple.relation,
            "tail": triple.tail,
            "head_type": triple.head_type,
            "tail_type": triple.tail_type,
            "attributes": triple.attributes,
            "provenance": triple.provenance
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-5"
) -> CTKGConstructionAgent:
    """
    Create a CTKG Construction Agent with default configuration.
    
    Args:
        api_key: API key for LLM service (optional, uses env var if not provided)
        model: Model name to use
        
    Returns:
        Configured CTKGConstructionAgent instance
    """
    import os
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    return CTKGConstructionAgent(
        api_key=api_key,
        model=model
    )


def process_trial(
    nct_id: str,
    tasks: Optional[List[str]] = None,
    ctkg_type: str = "trial_centric",
    api_key: Optional[str] = None
) -> Dict:
    """
    Convenience function to process a single trial.
    
    Args:
        nct_id: NCT identifier
        tasks: List of task names (optional)
        ctkg_type: Type of CTKG ("trial_centric", "intervention_centric", "dynamic")
        api_key: API key for LLM service
        
    Returns:
        Dictionary containing processed results
    """
    agent = create_agent(api_key=api_key)
    
    # Load trial from API
    agent.load_trial_from_api(nct_id)
    
    # Convert task names to TaskType
    task_types = None
    if tasks:
        task_types = [TaskType(f"task{i}_{t}") for i, t in enumerate(tasks, 1)]
    
    # Execute pipeline
    ctkg_type_enum = CTKGType(ctkg_type)
    results = agent.execute_pipeline(nct_id, tasks=task_types, ctkg_type=ctkg_type_enum)
    
    return {
        "nct_id": nct_id,
        "results": {t.value: r.data for t, r in results.items() if r.success}
    }


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
import time
import csv
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Reduce noisy transport/client logs in user-facing runs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


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
        model: str = "gpt-5-mini",
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        user_facing_logs: bool = True
    ):
        """
        Initialize the CTKG Construction Agent.
        
        Args:
            llm_client: Pre-configured LLM client (optional)
            api_key: API key for LLM service
            model: Model name to use (default: gpt-5-mini)
            cache_dir: Directory for caching intermediate results
        """
        self.llm_client = llm_client
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir or Path("./cache")
        self.output_dir = output_dir or (Path(__file__).resolve().parent / "output")
        self.user_facing_logs = user_facing_logs
        
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
        
        if self.user_facing_logs:
            logger.info(f"[Step 0] Looking up NCT ID: {nct_id} ...")

        raw_data = fetch_trial_by_nct_id(nct_id)
        trial_data = TrialData.from_raw(raw_data)
        self.trial_data_cache[nct_id] = trial_data
        
        logger.info(f"Loaded trial {nct_id} from ClinicalTrials.gov API")
        if self.user_facing_logs:
            logger.info(self._format_trial_overview(self._extract_trial_overview(trial_data)))
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
        
        if self.user_facing_logs:
            logger.info(f"[Step 0] Loading NCT ID: {nct_id} from file {file_path} ...")

        raw_data = load_trial_from_json(file_path, nct_id)
        trial_data = TrialData.from_raw(raw_data)
        self.trial_data_cache[nct_id] = trial_data
        
        logger.info(f"Loaded trial {nct_id} from {file_path}")
        if self.user_facing_logs:
            logger.info(self._format_trial_overview(self._extract_trial_overview(trial_data)))
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
        
        total = len(nct_ids)
        for i, nct_id in enumerate(nct_ids, start=1):
            try:
                if self.user_facing_logs:
                    logger.info(f"[Step 0] Processing {i}/{total}: {nct_id}")
                if source == "api":
                    results[nct_id] = self.load_trial_from_api(nct_id)
                elif source == "file" and file_path:
                    results[nct_id] = self.load_trial_from_file(file_path, nct_id)
            except Exception as e:
                logger.error(f"Failed to load trial {nct_id}: {e}")
                
        return results

    def _extract_trial_overview(self, trial_data: TrialData) -> Dict:
        """Extract user-facing trial overview for step-1 display."""
        conditions = trial_data.conditions.get("conditions", [])
        interventions = [
            i.get("name")
            for i in trial_data.arms_interventions.get("interventions", [])
            if i.get("name")
        ]
        start_date = trial_data.status.get("startDateStruct", {}).get("date")
        completion_date = trial_data.status.get("completionDateStruct", {}).get("date")

        outcome_measures = trial_data.outcome_measures.get("outcomeMeasures", [])
        analyses_count = 0
        for measure in outcome_measures:
            analyses_count += len(measure.get("analyses", []))

        return {
            "nct_id": trial_data.nct_id,
            "has_results": trial_data.has_results,
            "start_date": start_date or "N/A",
            "completion_date": completion_date or "N/A",
            "has_statistical_results": analyses_count > 0,
            "statistical_analyses_count": analyses_count,
            "conditions": conditions,
            "interventions": interventions,
        }

    def _format_trial_overview(self, overview: Dict) -> str:
        """Format step-1 overview into user-friendly English logs."""
        conditions = overview.get("conditions", [])
        interventions = overview.get("interventions", [])
        conditions_text = ", ".join(conditions) if conditions else "None reported"
        interventions_text = ", ".join(interventions) if interventions else "None reported"

        return (
            f"[Step 0] Trial Summary for {overview.get('nct_id', 'Unknown')}\n"
            f"  - Has posted results: {overview.get('has_results', False)}\n"
            f"  - Start date: {overview.get('start_date', 'N/A')}\n"
            f"  - Completion date: {overview.get('completion_date', 'N/A')}\n"
            f"  - Has statistical analyses: {overview.get('has_statistical_results', False)}\n"
            f"  - Statistical analyses count: {overview.get('statistical_analyses_count', 0)}\n"
            f"  - Conditions: {conditions_text}\n"
            f"  - Interventions: {interventions_text}"
        )
    
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

        # Hydrate upstream task_results for standalone dependent task execution.
        if task_type == TaskType.TASK5_STATISTICAL_CONCLUSIONS and "task_results" not in kwargs:
            hydrated = {}
            for upstream in [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING,
            ]:
                upstream_res = self._get_task_result(nct_id, upstream, use_disk_cache=True)
                if upstream_res is not None and upstream_res.success:
                    hydrated[upstream.value] = upstream_res.data or {}
            kwargs["task_results"] = hydrated

        if task_type == TaskType.TASK8_CTKG_ASSEMBLY and "task_results" not in kwargs:
            hydrated = {}
            for upstream in [
                TaskType.TASK1_OUTCOME_STANDARDIZATION,
                TaskType.TASK2_INTERVENTION_PROFILING,
                TaskType.TASK3_ELIGIBILITY_STRUCTURING,
                TaskType.TASK4_PURPOSE_INFERENCE,
                TaskType.TASK5_STATISTICAL_CONCLUSIONS,
                TaskType.TASK6_DISEASE_MAPPING,
                TaskType.TASK7_DYNAMIC_CTKG,
            ]:
                upstream_res = self._get_task_result(nct_id, upstream, use_disk_cache=True)
                if upstream_res is not None and upstream_res.success:
                    hydrated[upstream.value] = upstream_res.data or {}
            kwargs["task_results"] = hydrated
        
        # Execute task
        try:
            task_start = time.perf_counter()
            if self.user_facing_logs:
                logger.info(f"[Task] Running {task_type.value} for {nct_id} ... [timer started]")
            result = handler.execute(trial_data, **kwargs)
            task_elapsed = time.perf_counter() - task_start
            
            # Cache result
            if nct_id not in self.task_results:
                self.task_results[nct_id] = {}
            self.task_results[nct_id][task_type] = result
            self._save_task_result_to_disk(nct_id, task_type, result)
            
            logger.info(
                f"Completed {task_type.value} for {nct_id} "
                f"in {self._format_duration(task_elapsed)}"
            )
            if self.user_facing_logs:
                logger.info(self._format_task_result(nct_id, task_type, result))
                if task_type == TaskType.TASK1_OUTCOME_STANDARDIZATION:
                    self._log_task1_outcomes(result)
                if task_type == TaskType.TASK2_INTERVENTION_PROFILING:
                    self._log_task2_output(result)
                if task_type == TaskType.TASK3_ELIGIBILITY_STRUCTURING:
                    self._log_task3_output(result)
                if task_type == TaskType.TASK4_PURPOSE_INFERENCE:
                    self._log_task4_output(result)
                if task_type == TaskType.TASK5_STATISTICAL_CONCLUSIONS:
                    self._log_task_kg_triples("Task5", result)
                if task_type == TaskType.TASK6_DISEASE_MAPPING:
                    self._log_task_kg_triples("Task6", result)
                if task_type == TaskType.TASK8_CTKG_ASSEMBLY:
                    self._log_task8_output(result)
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
        ctkg_type: CTKGType = CTKGType.TRIAL_CENTRIC,
        reuse_cached_tasks: bool = True,
        force_rerun_tasks: Optional[List[TaskType]] = None,
        **kwargs
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
        force_rerun_set = set(force_rerun_tasks or [])
        if self.user_facing_logs:
            logger.info(
                f"[Agent] Starting {ctkg_type.value.replace('_', ' ')} pipeline for {nct_id} "
                f"({len(tasks)} step(s))."
            )
        
        results = {}
        pipeline_start = time.perf_counter()
        completed_durations: List[float] = []
        
        for idx, task_type in enumerate(tasks, start=1):
            if self.user_facing_logs:
                global_idx = self._task_global_index(task_type)
                logger.info(
                    f"[Step {global_idx}/9] {self._friendly_task_name(task_type)} started."
                )
            # Check dependencies
            dependencies = self._get_task_dependencies(task_type)
            dependencies_satisfied = True
            for dep in dependencies:
                dep_result = results.get(dep) or self._get_task_result(nct_id, dep, use_disk_cache=reuse_cached_tasks)

                if dep_result is None or not dep_result.success:
                    logger.warning(f"Skipping {task_type.value}: dependency {dep.value} not satisfied")
                    dependencies_satisfied = False
                    break
                # Make dependency data available to downstream task kwargs.
                if dep not in results:
                    results[dep] = dep_result
            if not dependencies_satisfied:
                results[task_type] = TaskResult(
                    task_type=task_type,
                    success=False,
                    errors=["Dependency not satisfied"],
                    metadata={"skipped": True}
                )
                if self.user_facing_logs:
                    global_idx = self._task_global_index(task_type)
                    logger.info(
                        f"[Step {global_idx}/9] {self._friendly_task_name(task_type)} skipped "
                        f"because a required previous step did not succeed."
                    )
                continue

            # Reuse cached task result if available and rerun not forced.
            if reuse_cached_tasks and task_type not in force_rerun_set:
                cached_result = self._get_task_result(nct_id, task_type, use_disk_cache=True)
                if cached_result is not None and cached_result.success:
                    results[task_type] = cached_result
                    self._save_task_result_to_debug(
                        nct_id,
                        task_type,
                        {
                            "task_type": task_type.value,
                            "success": cached_result.success,
                            "data": cached_result.data or {},
                            "errors": cached_result.errors or [],
                            "metadata": cached_result.metadata or {},
                        },
                    )
                    if self.user_facing_logs:
                        global_idx = self._task_global_index(task_type)
                        logger.info(
                            f"[Step {global_idx}/9] {self._friendly_task_name(task_type)} reused "
                            f"cached result ({nct_id}_{task_type.value}.json)."
                        )
                        logger.info(self._format_task_result(nct_id, task_type, cached_result))
                    continue
            
            # Execute task
            step_start = time.perf_counter()
            task_kwargs = self._build_task_kwargs(
                task_type,
                results,
                ctkg_type,
                nct_id=nct_id,
                **kwargs
            )
            result = self.execute_task(nct_id, task_type, **task_kwargs)
            results[task_type] = result
            step_elapsed = time.perf_counter() - step_start
            completed_durations.append(step_elapsed)
            if self.user_facing_logs:
                global_idx = self._task_global_index(task_type)
                logger.info(
                    f"[Step {global_idx}/9] {self._friendly_task_name(task_type)} finished "
                    f"in {self._format_duration(step_elapsed)}."
                )

        if self.user_facing_logs:
            success_count = sum(
                1 for task in tasks
                if task in results and results[task].success
            )
            pipeline_elapsed = time.perf_counter() - pipeline_start
            global_step_ids = [self._task_global_index(t) for t in tasks]
            global_step_ids_text = ", ".join(str(i) for i in global_step_ids if i > 0) or "N/A"
            logger.info(
                f"[Agent] Pipeline completed for {nct_id}. "
                f"Successful selected steps: {success_count}/{len(tasks)}. "
                f"Global step IDs run: [{global_step_ids_text}]. "
                f"Total time: {self._format_duration(pipeline_elapsed)}."
            )
        
        return results

    def _get_task_result(
        self,
        nct_id: str,
        task_type: TaskType,
        use_disk_cache: bool = True
    ) -> Optional[TaskResult]:
        """Fetch task result from memory first, then optional disk cache."""
        mem_result = self.task_results.get(nct_id, {}).get(task_type)
        if mem_result is not None:
            return mem_result
        if not use_disk_cache:
            return None
        return self._load_task_result_from_disk(nct_id, task_type)

    def _task_cache_file(self, nct_id: str, task_type: TaskType) -> Path:
        """Path for task result cache file: <cache>/<nct>/<nct>_<task>.json."""
        return self.cache_dir / nct_id / f"{nct_id}_{task_type.value}.json"

    def _task_output_file(self, nct_id: str, task_type: TaskType) -> Path:
        """Path for user-facing output file: <output>/<nct>/<nct>_<task>.json."""
        return self.output_dir / nct_id / f"{nct_id}_{task_type.value}.json"

    def _save_task_result_to_disk(self, nct_id: str, task_type: TaskType, result: TaskResult) -> None:
        """Persist task result to cache (reuse) and output (user-facing)."""
        payload = {
            "task_type": task_type.value,
            "success": result.success,
            "data": result.data or {},
            "errors": result.errors or [],
            "metadata": result.metadata or {},
        }
        # Cache output (used by dependency reuse)
        cache_path = self._task_cache_file(nct_id, task_type)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # User-facing output (fixed output directory)
        output_path = self._task_output_file(nct_id, task_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if task_type == TaskType.TASK8_CTKG_ASSEMBLY:
            # User requested Task8 to keep only the main JSON output.
            # Remove legacy HRT sidecar files if they exist.
            for p in [
                self.cache_dir / nct_id / f"{nct_id}_{task_type.value}_kg_triples_hrt.json",
                self.output_dir / nct_id / f"{nct_id}_{task_type.value}_kg_triples_hrt.json",
            ]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove legacy HRT file {p}: {e}")

        # Task7: also write user-facing CSV to output directory.
        if task_type == TaskType.TASK7_DYNAMIC_CTKG:
            csv_path = self.output_dir / nct_id / f"{nct_id}_{task_type.value}.csv"
            rows = ((payload.get("data") or {}).get("changes_table") or [])
            fieldnames = ["nct_id", "version", "field_name", "old", "new", "change_category"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "nct_id": row.get("nct_id", ""),
                        "version": row.get("version", ""),
                        "field_name": row.get("field_name", ""),
                        "old": row.get("old", ""),
                        "new": row.get("new", ""),
                        "change_category": row.get("change_category", ""),
                    })
        self._save_task_result_to_debug(nct_id, task_type, payload)

    def _save_task_result_to_debug(self, nct_id: str, task_type: TaskType, payload: Dict) -> None:
        """
        Persist task output to fixed debug path:
        agents/construction/debug/step{N+1}_task{N}/{NCTID}_task{N}_output.json
        where N is global task index (1..9).
        """
        task_idx = self._task_global_index(task_type)
        if task_idx <= 0:
            return
        debug_dir = Path("agents/construction/debug") / f"step{task_idx + 1}_task{task_idx}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        if task_type == TaskType.TASK7_DYNAMIC_CTKG:
            csv_file = debug_dir / f"{nct_id}_task{task_idx}_output.csv"
            rows = ((payload.get("data") or {}).get("changes_table") or [])
            fieldnames = ["nct_id", "version", "field_name", "old", "new", "change_category"]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "nct_id": row.get("nct_id", ""),
                        "version": row.get("version", ""),
                        "field_name": row.get("field_name", ""),
                        "old": row.get("old", ""),
                        "new": row.get("new", ""),
                        "change_category": row.get("change_category", ""),
                    })
            return

        if task_type == TaskType.TASK8_CTKG_ASSEMBLY:
            # Remove legacy debug HRT file for Task8.
            hrt_file = debug_dir / f"{nct_id}_task{task_idx}_kg_triples_hrt.json"
            try:
                if hrt_file.exists():
                    hrt_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove legacy debug HRT file {hrt_file}: {e}")

        debug_file = debug_dir / f"{nct_id}_task{task_idx}_output.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _extract_hrt_triples(self, payload: Dict) -> List[Dict]:
        """Extract pure (head_entity, relationship, tail_entity) triples from a task payload."""
        data = (payload or {}).get("data", {}) or {}
        hrt = data.get("kg_triples_hrt", [])
        if isinstance(hrt, list) and hrt:
            return hrt

        # Task8 stores graph triples under trial/intervention/dynamic lists;
        # other tasks may use `triples` or `kg_triples`.
        candidate_keys = [
            "trial_centric_triples",
            "intervention_centric_triples",
            "dynamic_ctkg_triples",
            "triples",
            "kg_triples",
        ]

        raw_triples: List[Dict] = []
        for key in candidate_keys:
            v = data.get(key)
            if isinstance(v, list):
                raw_triples.extend([x for x in v if isinstance(x, dict)])

        out: List[Dict] = []
        seen = set()
        for t in raw_triples:
            head = t.get("head", "")
            rel = t.get("relation", "")
            tail = t.get("tail", "")
            if isinstance(head, str) and isinstance(rel, str) and isinstance(tail, str) and head and rel and tail:
                key = (head, rel, tail)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "head_entity": head,
                    "relationship": rel,
                    "tail_entity": tail,
                })
        return out

    def _load_task_result_from_disk(self, nct_id: str, task_type: TaskType) -> Optional[TaskResult]:
        """Load task result from disk cache and hydrate memory cache."""
        path = self._task_cache_file(nct_id, task_type)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            result = TaskResult(
                task_type=task_type,
                success=bool(payload.get("success", False)),
                data=payload.get("data", {}) or {},
                errors=payload.get("errors", []) or [],
                metadata=payload.get("metadata", {}) or {},
            )
            if nct_id not in self.task_results:
                self.task_results[nct_id] = {}
            self.task_results[nct_id][task_type] = result
            return result
        except Exception as e:
            logger.warning(f"Failed to load cached result for {nct_id} {task_type.value}: {e}")
            return None

    def _format_task_result(self, nct_id: str, task_type: TaskType, result: TaskResult) -> str:
        """Create user-facing task result summary."""
        data = result.data or {}
        errors = result.errors or []

        key_pairs = [
            ("total_outcomes", "total_outcomes"),
            ("processed_outcomes", "processed_outcomes"),
            ("processed_outcome_titles", "processed_outcome_titles"),
            ("total_groups", "total_groups"),
            ("processed_groups", "processed_groups"),
            ("total_analyses", "total_analyses"),
            ("processed_analyses", "processed_analyses"),
            ("linked_count", "linked_count"),
            ("unlinked_count", "unlinked_count"),
            ("total_trial_centric", "trial_triples"),
            ("total_intervention_centric", "intervention_triples"),
            ("total_dynamic", "dynamic_triples"),
            ("total_versions", "total_versions"),
        ]

        metrics = []
        for key, label in key_pairs:
            if key in data:
                metrics.append(f"{label}={data.get(key)}")

        if "triples" in data and isinstance(data.get("triples"), list):
            metrics.append(f"triples={len(data.get('triples', []))}")
        if "kg_triples" in data and isinstance(data.get("kg_triples"), list):
            metrics.append(f"kg_triples={len(data.get('kg_triples', []))}")
        # Nested task-specific summary metrics
        structured_eligibility = data.get("structured_eligibility", {})
        if isinstance(structured_eligibility, dict):
            q = structured_eligibility.get("entity_qualifiers")
            if isinstance(q, list):
                metrics.append(f"entity_qualifiers={len(q)}")
            k = structured_eligibility.get("kg_triples")
            if isinstance(k, list):
                metrics.append(f"kg_triples={len(k)}")
        inferred_purpose = data.get("inferred_purpose", {})
        if isinstance(inferred_purpose, dict):
            k = inferred_purpose.get("kg_triples")
            if isinstance(k, list):
                metrics.append(f"kg_triples={len(k)}")
            if isinstance(inferred_purpose.get("purpose"), str):
                metrics.append(f"purpose={inferred_purpose.get('purpose')}")
        if "message" in data:
            metrics.append(f"message={data.get('message')}")

        status = "completed" if result.success else "failed"
        metric_text = ", ".join(metrics) if metrics else "no summary metrics"
        return (
            f"[Task Result] {self._friendly_task_name(task_type)} ({nct_id}): {status}; "
            f"{metric_text}; errors={len(errors)}"
        )

    def _friendly_task_name(self, task_type: TaskType) -> str:
        """Human-readable task name for product-facing logs."""
        mapping = {
            TaskType.TASK1_OUTCOME_STANDARDIZATION: "Outcome Standardization",
            TaskType.TASK2_INTERVENTION_PROFILING: "Intervention Profiling",
            TaskType.TASK3_ELIGIBILITY_STRUCTURING: "Eligibility Structuring",
            TaskType.TASK4_PURPOSE_INFERENCE: "Purpose Inference",
            TaskType.TASK5_STATISTICAL_CONCLUSIONS: "Statistical Conclusions",
            TaskType.TASK6_DISEASE_MAPPING: "Disease Mapping",
            TaskType.TASK7_DYNAMIC_CTKG: "Dynamic CTKG",
            TaskType.TASK8_CTKG_ASSEMBLY: "CTKG Assembly",
            TaskType.TASK9_ENTITY_LINKING: "Entity Linking",
        }
        return mapping.get(task_type, task_type.value)

    def _task_global_index(self, task_type: TaskType) -> int:
        """Return global 1-9 step index for a task."""
        order = [
            TaskType.TASK1_OUTCOME_STANDARDIZATION,
            TaskType.TASK2_INTERVENTION_PROFILING,
            TaskType.TASK3_ELIGIBILITY_STRUCTURING,
            TaskType.TASK4_PURPOSE_INFERENCE,
            TaskType.TASK5_STATISTICAL_CONCLUSIONS,
            TaskType.TASK6_DISEASE_MAPPING,
            TaskType.TASK7_DYNAMIC_CTKG,
            TaskType.TASK8_CTKG_ASSEMBLY,
            TaskType.TASK9_ENTITY_LINKING,
        ]
        try:
            return order.index(task_type) + 1
        except ValueError:
            return 0

    def _format_duration(self, seconds: float) -> str:
        """Format duration into a friendly string."""
        if seconds < 1:
            return "<1s"
        if seconds < 60:
            return f"{int(round(seconds))}s"
        minutes = int(seconds // 60)
        rem = int(round(seconds % 60))
        return f"{minutes}m {rem}s"

    def _estimate_remaining_seconds(self, completed: List[float], remaining_steps: int) -> Optional[float]:
        """Estimate remaining seconds using completed-step average."""
        if remaining_steps <= 0:
            return 0.0
        if not completed:
            return None
        avg = sum(completed) / len(completed)
        return avg * remaining_steps

    def _format_eta_seconds(self, eta_seconds: Optional[float]) -> str:
        """Format ETA string for logs."""
        if eta_seconds is None:
            return "calculating..."
        return self._format_duration(eta_seconds)

    def _log_task1_outcomes(self, result: TaskResult) -> None:
        """Print Task1 normalized outcomes in user-requested format."""
        outcomes = (result.data or {}).get("standardized_outcomes", [])
        if not outcomes:
            logger.info("[Task1 Output] No standardized outcomes generated.")
            return

        logger.info(f"[Task1 Output] Standardized outcome measurements: {len(outcomes)}")
        for idx, item in enumerate(outcomes, start=1):
            # Keep exact JSON structure for easy user inspection.
            logger.info(
                f"[Task1 Output {idx}/{len(outcomes)}]\n"
                f"{json.dumps(item, ensure_ascii=False, indent=2)}"
            )

    def _log_task2_output(self, result: TaskResult) -> None:
        """Print Task2 raw profiling output for end users."""
        data = result.data or {}
        profiled = data.get("profiled_interventions", [])
        co_treat = data.get("co_treat_relations", [])
        composite = data.get("composite_relations", [])

        logger.info(
            f"[Task2 Output] Profiled groups={len(profiled)}, "
            f"co_treat_relations={len(co_treat)}, composite_relations={len(composite)}"
        )

        if not profiled:
            logger.info("[Task2 Output] No profiled interventions generated.")
            return

        for idx, group in enumerate(profiled, start=1):
            logger.info(
                f"[Task2 Output Group {idx}/{len(profiled)}]\n"
                f"{json.dumps(group, ensure_ascii=False, indent=2)}"
            )

        if co_treat:
            logger.info(
                f"[Task2 Output] co_treat_relations\n"
                f"{json.dumps(co_treat, ensure_ascii=False, indent=2)}"
            )
        if composite:
            logger.info(
                f"[Task2 Output] composite_relations\n"
                f"{json.dumps(composite, ensure_ascii=False, indent=2)}"
            )

    def _log_task_kg_triples(self, task_label: str, result: TaskResult) -> None:
        """Print task kg_triples in full JSON for direct user inspection."""
        triples = (result.data or {}).get("kg_triples", [])
        if not triples:
            logger.info(f"[{task_label} Output] No kg_triples generated.")
            return

        logger.info(f"[{task_label} Output] Final kg_triples: {len(triples)}")
        for idx, triple in enumerate(triples, start=1):
            logger.info(
                f"[{task_label} Output Triple {idx}/{len(triples)}]\n"
                f"{json.dumps(triple, ensure_ascii=False, indent=2)}"
            )

    def _log_task3_output(self, result: TaskResult) -> None:
        """Print concise Task3 summary and a short kg triple preview."""
        se = (result.data or {}).get("structured_eligibility", {}) or {}
        qualifiers = se.get("entity_qualifiers", []) or []
        triples = se.get("kg_triples", []) or []
        logger.info(
            f"[Task3 Output] entity_qualifiers={len(qualifiers)}, kg_triples={len(triples)}"
        )
        for idx, triple in enumerate(triples[:3], start=1):
            logger.info(
                f"[Task3 Triple Preview {idx}/3]\n"
                f"{json.dumps(triple, ensure_ascii=False, indent=2)}"
            )

    def _log_task4_output(self, result: TaskResult) -> None:
        """Print concise Task4 summary and a short kg triple preview."""
        inferred = (result.data or {}).get("inferred_purpose", {}) or {}
        purpose = inferred.get("purpose")
        answer = inferred.get("answer")
        triples = inferred.get("kg_triples", []) or []
        logger.info(
            f"[Task4 Output] answer={answer}, purpose={purpose}, kg_triples={len(triples)}"
        )
        for idx, triple in enumerate(triples[:3], start=1):
            logger.info(
                f"[Task4 Triple Preview {idx}/3]\n"
                f"{json.dumps(triple, ensure_ascii=False, indent=2)}"
            )

    def _log_task8_output(self, result: TaskResult) -> None:
        """Print final assembled triples summary and short previews."""
        data = result.data or {}
        trial = data.get("trial_centric_triples", []) or []
        inter = data.get("intervention_centric_triples", []) or []
        dyn = data.get("dynamic_ctkg_triples", []) or []
        logger.info(
            f"[Task8 Output] trial={len(trial)}, intervention={len(inter)}, dynamic={len(dyn)}"
        )
        for idx, triple in enumerate(trial[:3], start=1):
            logger.info(
                f"[Task8 Trial Preview {idx}/3]\n"
                f"{json.dumps(triple, ensure_ascii=False, indent=2)}"
            )
        for idx, triple in enumerate(inter[:3], start=1):
            logger.info(
                f"[Task8 Intervention Preview {idx}/3]\n"
                f"{json.dumps(triple, ensure_ascii=False, indent=2)}"
            )

    def _build_task_kwargs(
        self,
        task_type: TaskType,
        results: Dict[TaskType, TaskResult],
        ctkg_type: CTKGType,
        nct_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Build task-specific kwargs from upstream results."""
        result_data = {
            t.value: r.data for t, r in results.items() if r.success
        }
        task_kwargs = {"task_results": result_data}
        task_kwargs.update(kwargs)

        if task_type == TaskType.TASK6_DISEASE_MAPPING:
            task5_key = TaskType.TASK5_STATISTICAL_CONCLUSIONS.value
            task_kwargs["task5_results"] = result_data.get(task5_key, {})

        if task_type == TaskType.TASK8_CTKG_ASSEMBLY:
            # Task 8 consumes outputs from multiple upstream tasks. When users run
            # Task 8 directly, proactively hydrate missing upstream results from
            # memory/disk cache to keep assembly behavior consistent.
            if nct_id:
                upstream_tasks = [
                    TaskType.TASK1_OUTCOME_STANDARDIZATION,
                    TaskType.TASK2_INTERVENTION_PROFILING,
                    TaskType.TASK3_ELIGIBILITY_STRUCTURING,
                    TaskType.TASK4_PURPOSE_INFERENCE,
                    TaskType.TASK5_STATISTICAL_CONCLUSIONS,
                    TaskType.TASK6_DISEASE_MAPPING,
                    TaskType.TASK7_DYNAMIC_CTKG,
                ]
                for upstream_task in upstream_tasks:
                    key = upstream_task.value
                    if key in result_data:
                        continue
                    cached_result = self._get_task_result(nct_id, upstream_task, use_disk_cache=True)
                    if cached_result is not None and cached_result.success:
                        result_data[key] = cached_result.data or {}
                task_kwargs["task_results"] = result_data

            if ctkg_type == CTKGType.TRIAL_CENTRIC:
                task_kwargs["ctkg_type"] = "trial_centric"
                if "include_dynamic" not in kwargs:
                    task_kwargs["include_dynamic"] = True
            elif ctkg_type == CTKGType.INTERVENTION_CENTRIC:
                task_kwargs["ctkg_type"] = "intervention_centric"
                if "include_dynamic" not in kwargs:
                    task_kwargs["include_dynamic"] = False
            else:
                task_kwargs["ctkg_type"] = "both"
                if "include_dynamic" not in kwargs:
                    task_kwargs["include_dynamic"] = True

        return task_kwargs
    
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
            TaskType.TASK9_ENTITY_LINKING: [],
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
        return self._extract_triples_from_pipeline(results, CTKGType.TRIAL_CENTRIC)
    
    def generate_intervention_centric_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """
        Generate intervention-centric CTKG for a given trial.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of CTKGTriple objects representing the knowledge graph
        """
        results = self.execute_pipeline(nct_id, ctkg_type=CTKGType.INTERVENTION_CENTRIC)
        return self._extract_triples_from_pipeline(results, CTKGType.INTERVENTION_CENTRIC)
    
    def generate_dynamic_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """
        Generate dynamic CTKG (version history) for a given trial.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            List of CTKGTriple objects representing version changes
        """
        results = self.execute_pipeline(nct_id, ctkg_type=CTKGType.DYNAMIC)
        return self._extract_triples_from_pipeline(results, CTKGType.DYNAMIC)

    def generate_ctkg_batch(
        self,
        nct_ids: List[str],
        ctkg_type: CTKGType = CTKGType.TRIAL_CENTRIC,
        source: str = "api",
        file_path: Optional[Path] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate CTKG for one or more NCT IDs with graceful task skipping.
        """
        self.load_trials_batch(nct_ids, source=source, file_path=file_path)
        batch_results: Dict[str, List[Dict]] = {}

        for nct_id in nct_ids:
            if nct_id not in self.trial_data_cache:
                continue

            try:
                pipeline_results = self.execute_pipeline(nct_id, ctkg_type=ctkg_type)
                batch_results[nct_id] = self._extract_triples_from_pipeline(
                    pipeline_results, ctkg_type
                )
            except Exception as e:
                logger.error(f"Failed to generate CTKG for {nct_id}: {e}")
                batch_results[nct_id] = []

        return batch_results

    def run_pipeline_with_structured_output(
        self,
        nct_id: str,
        tasks: Optional[List[TaskType]] = None,
        ctkg_type: CTKGType = CTKGType.TRIAL_CENTRIC,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict:
        """
        Run pipeline and return a product-facing structured JSON object.
        Optionally writes the object to output_path.
        """
        if nct_id not in self.trial_data_cache:
            self.load_trial_from_api(nct_id)

        selected_tasks = tasks or self._get_default_tasks_for_ctkg_type(ctkg_type)
        results = self.execute_pipeline(
            nct_id=nct_id,
            tasks=selected_tasks,
            ctkg_type=ctkg_type,
            **kwargs
        )

        triples = self._extract_triples_from_pipeline(results, ctkg_type)
        trial_data = self.trial_data_cache[nct_id]
        overview = self._extract_trial_overview(trial_data)

        task_summaries = []
        for task in selected_tasks:
            r = results.get(task)
            if r is None:
                continue
            task_summaries.append({
                "task": task.value,
                "success": bool(r.success),
                "skipped": bool((r.metadata or {}).get("skipped", False)),
                "error_count": len(r.errors or []),
                "message": (r.data or {}).get("message"),
            })

        output = {
            "nct_id": nct_id,
            "ctkg_type": ctkg_type.value,
            "trial_overview": overview,
            "pipeline_summary": {
                "total_tasks": len(selected_tasks),
                "successful_tasks": sum(1 for t in task_summaries if t.get("success")),
                "task_summaries": task_summaries,
            },
            "triples": triples,
            "triple_count": len(triples),
        }

        if output_path is not None:
            path = Path(output_path)
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"[Output] Structured JSON saved to {path}")

        return output

    def _extract_triples_from_pipeline(
        self,
        results: Dict[TaskType, TaskResult],
        ctkg_type: CTKGType
    ) -> List[Dict]:
        """Extract final triples from pipeline results by CTKG type."""
        if ctkg_type == CTKGType.DYNAMIC:
            if TaskType.TASK7_DYNAMIC_CTKG in results and results[TaskType.TASK7_DYNAMIC_CTKG].success:
                return results[TaskType.TASK7_DYNAMIC_CTKG].data.get('triples', [])
            return []

        if TaskType.TASK8_CTKG_ASSEMBLY not in results or not results[TaskType.TASK8_CTKG_ASSEMBLY].success:
            return []

        data = results[TaskType.TASK8_CTKG_ASSEMBLY].data
        if ctkg_type == CTKGType.TRIAL_CENTRIC:
            return data.get('trial_centric_triples', data.get('triples', []))
        if ctkg_type == CTKGType.INTERVENTION_CENTRIC:
            return data.get('intervention_centric_triples', data.get('triples', []))
        return data.get('triples', [])
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_ctkg(
        self,
        triples: List[Union[CTKGTriple, Dict]],
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
    
    def _triple_to_dict(self, triple: Union[CTKGTriple, Dict]) -> Dict:
        """Convert CTKGTriple to dictionary."""
        if isinstance(triple, dict):
            return triple
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
    model: str = "gpt-5-mini",
    output_dir: Optional[Path] = None
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
        model=model,
        output_dir=output_dir
    )


def create_api_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini"
) -> CTKGConstructionAgent:
    """Create an agent using OpenAI API backend."""
    return create_agent(api_key=api_key, model=model)


def create_endpoint_agent(
    base_url: str,
    model: str,
    api_key: str = "local-key"
) -> CTKGConstructionAgent:
    """
    Create an agent using an OpenAI-compatible endpoint (e.g., vLLM/TGI gateway).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package required. Install with: pip install openai") from e

    client = OpenAI(base_url=base_url, api_key=api_key)
    return CTKGConstructionAgent(llm_client=client, model=model)


def create_local_agent(
    local_model_path: str,
    model: Optional[str] = None,
    hf_kwargs: Optional[Dict] = None
) -> CTKGConstructionAgent:
    """
    Create an agent using a local Hugging Face model backend.

    Args:
        local_model_path: Local model path or HF model id.
        model: Logical model name exposed to handlers. Defaults to local_model_path.
        hf_kwargs: Optional kwargs for LocalHFChatClient (device, max_new_tokens, etc.).
    """
    from .model_clients import LocalHFChatClient

    hf_kwargs = hf_kwargs or {}
    client = LocalHFChatClient(model_path=local_model_path, **hf_kwargs)
    return CTKGConstructionAgent(
        llm_client=client,
        model=model or local_model_path
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
    task_types: Optional[List[TaskType]] = None
    if tasks:
        task_types = []
        for task_name in tasks:
            if isinstance(task_name, TaskType):
                task_types.append(task_name)
                continue

            try:
                task_types.append(TaskType(task_name))
                continue
            except Exception:
                pass

            matched = None
            for t in TaskType:
                if t.value == task_name or t.name.lower() == str(task_name).lower():
                    matched = t
                    break
            if matched:
                task_types.append(matched)
            else:
                raise ValueError(f"Unknown task: {task_name}")
    
    # Execute pipeline
    ctkg_type_enum = CTKGType(ctkg_type)
    results = agent.execute_pipeline(nct_id, tasks=task_types, ctkg_type=ctkg_type_enum)
    
    return {
        "nct_id": nct_id,
        "results": {t.value: r.data for t, r in results.items() if r.success}
    }

# CTKG Task I/O Specification

This document defines runtime input/output contracts for `Task1` to `Task9` in `agents/construction/tasks/`.

## Global

- All tasks implement `execute(trial_data: Any, **kwargs) -> TaskResult`.
- `TaskResult` returns payload in `result.data`.
- Cache output path: `cache/<NCT_ID>/<NCT_ID>_<task_name>.json`.
- Debug output path: `agents/construction/debug/step{N+1}_task{N}/<NCT_ID>_task{N}_output.json`.

## Task1 `task1_outcome_standardization`

### Input
- `trial_data`
- `kwargs`: none required

### Output `data`
- `standardized_outcomes: List[Dict]`
  - `original_title: str`
  - `core_measurement: str`
  - `attributes: Dict`
    - `outcome_type: str`
    - `measurement_tool: str`
    - `value_condition: str`
    - `conditional_population: str`
    - `paramType: Optional[str]`
    - `timeFrame: Optional[str]`
    - `unitOfMeasure: Optional[str]`
    - `units: Optional[str]`
    - `description: Optional[str]`
- `total_outcomes: int`
- `processed_outcomes: int`
- `errors: List[str]`
- Optional when empty:
  - `message: str`

## Task2 `task2_intervention_profiling`

### Input
- `trial_data`
- `kwargs`: none required

### Output `data`
- `profiled_interventions: List[Dict]`
  - `group_id: str`
  - `original_group_id: Optional[str]`
  - `group_title: str`
  - `original_description: str`
  - `interventions: List[Dict]`
  - `administration_sequence: str`
- `co_treat_relations: List[Dict]`
- `composite_relations: List[Dict]`
- `total_groups: int`
- `processed_groups: int`
- `data_source: str`
- `errors: List[str]`
- Optional when empty:
  - `message: str`

## Task3 `task3_eligibility_structuring`

### Input
- `trial_data`
- `kwargs`: none required

### Output `data`
- `structured_eligibility: Dict`
  - `metadata: Dict`
    - `healthy_volunteers: Optional[bool]`
    - `sex: Optional[str]`
    - `minimum_age: Optional[str]`
    - `maximum_age: Optional[str]`
    - `std_ages: List[str]`
  - `entity_qualifiers: List[Dict]` (primary normalized output)
    - `section: "Inclusion" | "Exclusion"`
    - `slot: "Medical conditions" | "Medications or therapy" | "Laboratory values" | "Other" | "Pregnancy" | "Demographics"`
    - `original_text: str`
    - `core_entity: str`
    - `qualifiers: Dict`
      - `polarity: str`
      - `entity_type: str`
      - `status: str`
      - `temporal: Optional[Dict]`
      - `numeric_constraint: Optional[Dict]`
      - `severity_grade: Optional[str]`
      - `prior_treatment: Optional[Dict]`
      - `conditional_clause: Optional[str]`
      - `evidence_or_rule_source: Optional[str]`
  - Optional: `baseline_characteristics: List[Dict]`
    - `title: Optional[str]`
    - `param_type: Optional[str]`
    - `unit: Optional[str]`
    - `categories: List[Dict]`
  - `kg_triples: List[Dict]`
    - Common triple shape:
      - `head: str`
      - `relation: str`
      - `tail: str`
      - `head_type: str`
      - `tail_type: str`
      - `tail_attributes: Dict`
    - For eligibility entity triples (`includes_*`, `excludes_*`):
      - `tail_attributes` = qualifier payload from `entity_qualifiers.qualifiers`
      - keys:
        - `polarity: str`
        - `entity_type: str`
        - `status: str`
        - `temporal: Optional[Dict]`
        - `numeric_constraint: Optional[Dict]`
        - `severity_grade: Optional[str]`
        - `prior_treatment: Optional[Dict]`
        - `conditional_clause: Optional[str]`
        - `evidence_or_rule_source: Optional[str]`
    - For baseline triples (`baseline_characteristics_*`):
      - `tail_attributes` keys:
        - `unit: Optional[str]`
        - `param_type: Optional[str]`
        - `title: Optional[str]`
- Optional when empty:
  - `message: str`

## Task4 `task4_purpose_inference`

### Input
- `trial_data`
- `kwargs`: none required

### Output `data`
- `inferred_purpose: Dict`
  - Core fields:
    - `answer: "yes" | "no"`
    - `reason: str`
    - `purpose: "treat" | "improve" | "support" | "decrease" | "increase" | "analysis"`
    - `head_entities: List[str]`
    - `tail_entity: List[str]`
    - `original_purpose: Optional[str]`
    - `nct_id: str`
  - `kg_triples: List[Dict]`
    - Purpose relation triples (`Intervention -> Purpose -> Target`):
      - `head: str` (intervention concept)
      - `relation: "treat" | "improve" | "support" | "decrease" | "increase" | "analysis"`
      - `tail: str` (disease/symptom/endpoint concept)
      - `head_type: "Intervention"`
      - `tail_type: "Disease/Symptom/Endpoint"`
      - `provenance: Dict`
        - `nct_id: str`

## Task5 `task5_statistical_conclusions`

### Input
- `trial_data`
- `kwargs`:
  - `task_results: Dict` (optional, used for richer KG triple mapping)

### Output `data`
- `conclusions: List[Dict]` (raw per-analysis interpretation records)
- `comparative_conclusions: List[Dict]`
- `non_comparative_conclusions: List[Dict]`
- `kg_triples: List[Dict]`
  - Triple schema:
    - `head_entity_1: str` (normalized group id, e.g. `NCTXXXXXXX_000`)
    - `head_entity_2: Optional[str]` (comparator group id)
    - `relation: Dict`
      - `comparative_relation: Optional[str]`
      - `context: Optional[str]`
    - `tail: str` (Task1 `core_measurement` when available; otherwise original outcome title)
    - `head_entity_1_type: "Group"`
    - `head_entity_2_type: Optional["Group"]`
    - `tail_type: "Outcome"`
    - `head_entity_1_attributes: Optional[Dict]` (Task2 profiled group object)
    - `head_entity_2_attributes: Optional[Dict]` (Task2 profiled comparator group object)
    - `tail_attributes: List[Dict]` (Task1 mapped outcome records; each item includes `original_title`, `core_measurement`, `attributes`)
    - `relation_attributes: Dict`
      - `is_significant: bool`
      - `p_value: Optional[str]`
      - `statistical_conclusion: Optional[str]`
      - `statistical_analysis: Dict`
      - `group_values: Dict`
    - `provenance: Dict`
      - `nct_id: str`
- `total_analyses: int`
- `processed_analyses: int`
- `errors: List[str]`
- Optional when empty:
  - `message: str`

## Task6 `task6_disease_mapping`

### Input
- `trial_data`
- `kwargs`:
  - `task5_results: Dict` (required)

### Output `data`
- `disease_relations: List[Dict]` (legacy flattened view)
- `kg_triples: List[Dict]`
  - Triple schema:
    - `head_entity_1: str` (group id)
    - `head_entity_2: Optional[str]` (comparator group id)
    - `relation: Dict`
      - `comparative_relation: Optional[str]`
      - `context: "treat" | "improve" | "alleviate" | "prevent" | "reduce_risk" | "no_clinical_effect" | "worsen"`
    - `tail: str` (inferred disease/symptom concept)
    - `head_entity_1_type: "Group"`
    - `head_entity_2_type: Optional["Group"]`
    - `tail_type: "Disease/symptoms"`
    - `head_entity_1_attributes: Dict` (Task2 group object)
    - `head_entity_2_attributes: Dict` (Task2 comparator group object)
    - `tail_attributes: Dict`
      - `linked_outcomes: List[str]`
    - `relation_attributes: Dict`
      - `evidence_quality: Dict`
        - `strength: Optional[str]`
        - `limitations: List[str]`
      - `reasoning: Optional[str]`
    - `provenance: Dict`
      - `nct_id: str`
- `inference_details: List[Dict]`
  - each item includes:
    - `head_pair: List[str]`
    - `inferred_conditions: List[Dict]`
    - `is_same_as_registered: Optional[bool]`
    - `clinical_interpretation: Optional[str]`
- `summary: Dict`
  - `task5_total_triples: int`
  - `task5_significant_triples: int`
  - `head_pair_groups: int`
  - `clinically_meaningful_relations: int`
  - `unique_inferred_diseases: int`
- `errors: List[str]`
- Optional when skipped:
  - `message: str`

## Task7 `task7_dynamic_ctkg`

### Input
- `trial_data`
- `kwargs`:
  - `version_history: List[Dict]` (optional)
  - `raw_version_data: Dict` (optional)
  - `structure_text: bool` (optional, default behavior enabled)

### Output `data`
- `version_nodes: List[Dict]`
- `change_relations: List[Dict]`
- `cthist_changes: List[Dict]`
- `changes_table: List[Dict]`
- `triples: List[Dict]`
- `total_versions: int`
- `first_version_date: Optional[str]`
- `last_version_date: Optional[str]`
- `termination_info: Optional[Dict]`
- `summary: Dict`
- Optional when empty/single version:
  - `message: str`

## Task8 `task8_ctkg_assembly`

### Input
- `trial_data`
- `kwargs`:
  - `task_results: Dict` (Task1-Task7 outputs)
  - `ctkg_type: str` (`trial_centric` | `intervention_centric` | `both`)
  - `include_dynamic: bool`

### Output `data`
- `trial_centric_ctkg: Optional[Dict]`
  - when present:
    - `triples: List[Dict]`
    - `version_history: Dict`
- `trial_centric_triples: List[Dict]`
- `intervention_centric_triples: List[Dict]`
- `dynamic_ctkg_triples: List[Dict]`
- `total_trial_centric: int`
- `total_intervention_centric: int`
- `total_dynamic: int`

### Triple Bucketing Rule
- `head == NCT_ID` -> trial-level (`trial_centric_triples`)
- `head != NCT_ID` (including `NCT_ID_XXX` group heads) -> intervention-level (`intervention_centric_triples`)

### Triple Schema (Task8 output lists)
- `head: str`
- `relation: str`
- `tail: str`
- `head_type: str`
- `tail_type: str`
- Optional: `tail_attributes: Dict`
- `provenance: Dict`
  - always includes:
    - `nct_id: str`

### Notes
- Task8 no longer emits a separate `*_kg_triples_hrt.json` sidecar file.
- Task3/Task4 `kg_triples` are merged into Task8 output.
- Task5/Task6 `kg_triples` are merged into intervention-level triples.

## Task9 `task9_entity_linking`

### Input
- `trial_data`
- `kwargs`:
  - `task_results: Dict`
  - `linking_config: Dict`

### Output `data`
- `linked_entities: List[Dict]`
- `unlinked_entities: List[Dict]`
- `clusters: List[Dict]`
- `total_entities: int`
- `linked_count: int`
- `unlinked_count: int`
- `cluster_count: int`
- Optional when empty:
  - `message: str`

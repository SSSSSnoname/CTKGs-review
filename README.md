# CTKG Construction Agent

This module builds a **Clinical Trial Knowledge Graphs (CTKGs)** from ClinicalTrials.gov records.
It is designed for reproducible extraction of trial-level and intervention-level triples, and is the construction backend used in this repository.

## What This Repository Part Does

Given one NCT trial record, the pipeline:

1. standardizes outcomes,
2. profiles interventions and arm structure,
3. structures eligibility criteria,
4. infers purpose relations,
5. extracts statistical comparative conclusions,
6. maps significant findings to disease-level effects,
7. captures longitudinal protocol changes,
8. assembles all outputs into final CTKG triples.

The resulting graph is split into:

- `trial_centric_triples`: trial-level facts (`head == NCT_ID`)
- `intervention_centric_triples`: group/intervention-level facts (`head != NCT_ID`)

## Directory Structure

```text
agents/construction/
├── ctkg_agent.py
├── TASK_IO_SPEC.md                 # full field-level I/O spec
├── output/                         # user-facing outputs
├── cache/                          # reusable intermediate cache
├── debug/                          # step-by-step debug artifacts
└── tasks/
    ├── task1_outcome.py
    ├── task2_intervention.py
    ├── task3_eligibility.py
    ├── task4_purpose.py
    ├── task5_statistical.py
    ├── task6_disease.py
    ├── task7_dynamic.py
    ├── task8_assembly.py
    └── task9_linking.py
```

## Quick Start

### Requirements

- Python 3.10+
- `OPENAI_API_KEY` set in environment

### Install

```bash
pip install openai requests
```

### Minimal end-to-end example

```python
from agents.construction import create_agent, TaskType

nct_id = "NCT02119676"
agent = create_agent(model="gpt-5-mini")

agent.load_trial_from_api(nct_id)

# Run tasks 1-8 in order (recommended for reproducibility)
for task in [
    TaskType.TASK1_OUTCOME_STANDARDIZATION,
    TaskType.TASK2_INTERVENTION_PROFILING,
    TaskType.TASK3_ELIGIBILITY_STRUCTURING,
    TaskType.TASK4_PURPOSE_INFERENCE,
    TaskType.TASK5_STATISTICAL_CONCLUSIONS,
    TaskType.TASK6_DISEASE_MAPPING,
    TaskType.TASK7_DYNAMIC_CTKG,
    TaskType.TASK8_CTKG_ASSEMBLY,
]:
    agent.execute_task(nct_id, task)
```

Generated user-facing outputs are written to:

```text
agents/construction/output/<NCT_ID>/<NCT_ID>_<task_name>.json
```

## Task Overview (1-8)

- `Task1` Outcome standardization
  - one record per standardized outcome core:
    - `original_title`, `core_measurement`, `attributes`

- `Task2` Intervention profiling
  - structured arm/group interventions
  - `co_treat_relations` and `composite_relations`
  - entity attributes are explicit (`head_attributes`, `tail_attributes`)

- `Task3` Eligibility structuring
  - normalized `entity_qualifiers`
  - eligibility triples in `structured_eligibility.kg_triples`

- `Task4` Purpose inference
  - purpose relation inference + `kg_triples`

- `Task5` Statistical conclusions
  - comparative statistical triples in two-head schema (`kg_triples`)

- `Task6` Disease mapping
  - disease-level relation triples derived from Task5 significant evidence

- `Task7` Dynamic CTKG
  - version-to-version change extraction, plus CSV change table export

- `Task8` CTKG assembly
  - merges Task1-7 outputs into final `trial_centric_triples` and `intervention_centric_triples`

## Task8 Merge Policy (Current)

### Bucketing

- `head == NCT_ID` -> trial level
- `head != NCT_ID` -> intervention level

### Inputs merged by Task8

- Trial level:
  - native trial metadata triples
  - Task3 `kg_triples` (raw passthrough)
  - Task4 `kg_triples` (raw passthrough)

- Intervention level:
  - Task2 `co_treat_relations` and `composite_relations` (raw passthrough)
  - Task5 `kg_triples` (raw passthrough)
  - Task6 `kg_triples` (raw passthrough)

### Output

- Main file:
  - `agents/construction/output/<NCT_ID>/<NCT_ID>_task8_ctkg_assembly.json`


## Output Locations

- User-facing outputs:
  - `agents/construction/output/<NCT_ID>/...`
- Reuse cache:
  - `cache/<NCT_ID>/...`
- Debug artifacts:
  - `agents/construction/debug/...`

## Documentation

For exact runtime field definitions for every task, use:

- `agents/construction/TASK_IO_SPEC.md`

This README is a usage and orientation guide. `TASK_IO_SPEC.md` is the strict schema reference.

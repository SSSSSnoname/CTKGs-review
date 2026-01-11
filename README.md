# CTKG Construction Agent

A modular AI agent that converts ClinicalTrials.gov records into structured Clinical Trial Knowledge Graphs (CTKGs).

## Overview

The CTKG Construction Agent is designed to automatically extract, standardize, and structure clinical trial information from ClinicalTrials.gov into knowledge graph triples. It uses GPT-4o as the controller to orchestrate 9 modular tasks.

## Architecture

```
agents/construction/
├── ctkg_agent.py          # Main agent class and orchestration
├── __init__.py            # Package exports
├── data_loader/           # Data loading utilities
│   ├── ctgov_api.py       # ClinicalTrials.gov API client
│   ├── cthist_downloader.py  # Version history downloader (R integration)
│   ├── download_version_history.R  # R script for cthist
│   └── file_loader.py     # Local file loading
└── tasks/                 # Individual task implementations
    ├── base.py            # Base task handler class
    ├── task1_outcome.py   # Outcome standardization
    ├── task2_intervention.py  # Intervention profiling
    ├── task3_eligibility.py   # Eligibility structuring
    ├── task4_purpose.py   # Purpose inference
    ├── task5_statistical.py   # Statistical conclusions
    ├── task6_disease.py   # Disease mapping
    ├── task7_dynamic.py   # Dynamic CTKG (version tracking)
    ├── task8_assembly.py  # CTKG assembly
    └── task9_linking.py   # Entity linking
```

## Tasks

| Task | Name | Description |
|------|------|-------------|
| Task 1 | Outcome Standardization | Decompose outcome measures into core indicators and attributes |
| Task 2 | Intervention Profiling | Extract intervention details (dosage, route, frequency, etc.) |
| Task 3 | Eligibility Structuring | Structure eligibility criteria (conditions, medications, pregnancy) |
| Task 4 | Purpose Inference | Infer intervention-disease relationships and study purpose |
| Task 5 | Statistical Conclusions | Extract statistical analysis conclusions and effect directions |
| Task 6 | Disease Mapping | Map outcome-level findings to disease-level effects |
| Task 7 | Dynamic CTKG | Track protocol version change history |
| Task 8 | CTKG Assembly | Assemble complete knowledge graph triples |
| Task 9 | Entity Linking | Link entities to standard ontologies (UMLS, etc.) |

## Quick Start

### 1. Installation

```bash
pip install openai requests
```

### 2. Basic Usage

```python
from agents.construction import CTKGConstructionAgent, create_agent

# Create agent
agent = create_agent(api_key="your-openai-api-key")

# Load trial from API
trial_data = agent.load_trial_from_api("NCT02119676")

# Generate trial-centric CTKG
triples = agent.generate_trial_centric_ctkg("NCT02119676")

# Export to JSON
agent.export_ctkg(triples, "output.json")
```

### 3. Using the Pipeline

```python
from agents.construction import CTKGConstructionAgent, TaskType, CTKGType

agent = CTKGConstructionAgent(api_key="your-api-key")
agent.load_trial_from_api("NCT02119676")

# Execute all tasks for trial-centric CTKG
results = agent.execute_pipeline("NCT02119676", ctkg_type=CTKGType.TRIAL_CENTRIC)

# Or execute specific tasks
results = agent.execute_pipeline("NCT02119676", tasks=[
    TaskType.TASK1_OUTCOME_STANDARDIZATION,
    TaskType.TASK2_INTERVENTION_PROFILING,
    TaskType.TASK3_ELIGIBILITY_STRUCTURING
])
```

## CTKG Types

The agent can generate three types of knowledge graphs:

### 1. Trial-Centric CTKG
Focus on trial metadata and design:
- Phase, status, conditions
- Interventions and arm groups
- Eligibility criteria
- Outcomes and adverse events

### 2. Intervention-Centric CTKG
Focus on intervention-outcome relationships:
- Statistical conclusions
- Effect directions
- Disease-level mappings

### 3. Dynamic CTKG
Focus on protocol changes over time:
- Version history
- Field-level changes
- Change status tracking (added/removed/modified)

## Output Format

### Triple Structure

```json
{
  "head": "NCT02119676",
  "relation": "hasIntervention",
  "tail": "Ruxolitinib",
  "head_type": "Trial",
  "tail_type": "Drug",
  "attributes": {
    "dosage": "5 mg",
    "frequency": "twice daily",
    "route": "oral"
  },
  "provenance": {
    "task": "task2",
    "source": "resultsSection"
  }
}
```

### Relation Types

| Category | Relations |
|----------|-----------|
| Trial Structure | `hasPhase`, `hasStatus`, `hasCondition`, `hasIntervention`, `hasOutcome` |
| Eligibility | `includes_condition`, `excludes_condition`, `includes_medication`, `excludes_medication` |
| Intervention | `treat`, `improve`, `decrease`, `increase`, `co_treat`, `composite` |
| Statistical | `outperforms`, `comparable_to`, `non_inferior_to` |
| Adverse Events | `hasSeriousAdverseEvent`, `hasOtherAdverseEvent`, `belongsToOrganSystem` |
| Baseline | `hasFemale`, `hasMale`, `hasAge_Continuous` |

## API Reference

### CTKGConstructionAgent

```python
class CTKGConstructionAgent:
    def __init__(self, llm_client=None, api_key=None, model="gpt-4o", cache_dir=None):
        """Initialize the CTKG Construction Agent."""
    
    def load_trial_from_api(self, nct_id: str) -> TrialData:
        """Download trial data from ClinicalTrials.gov API."""
    
    def load_trial_from_file(self, file_path, nct_id: str) -> TrialData:
        """Load trial data from local JSON file."""
    
    def execute_task(self, nct_id: str, task_type: TaskType, **kwargs) -> TaskResult:
        """Execute a single task for a given trial."""
    
    def execute_pipeline(self, nct_id: str, tasks=None, ctkg_type=CTKGType.TRIAL_CENTRIC) -> Dict:
        """Execute a pipeline of tasks."""
    
    def generate_trial_centric_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """Generate trial-centric CTKG."""
    
    def generate_intervention_centric_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """Generate intervention-centric CTKG."""
    
    def generate_dynamic_ctkg(self, nct_id: str) -> List[CTKGTriple]:
        """Generate dynamic CTKG with version tracking."""
    
    def export_ctkg(self, triples, output_path, format="json"):
        """Export CTKG to file."""
```

## Task Dependencies

```
Task 1 (Outcome) ─────┬─────> Task 5 (Statistical) ───> Task 6 (Disease)
Task 2 (Intervention) ─┘
Task 3 (Eligibility) ─────> (standalone)
Task 4 (Purpose) ─────────> (standalone)
Task 7 (Dynamic) ─────────> (standalone)
Task 8 (Assembly) <────────── Task 1, 2, 3, 4, 5, 6
Task 9 (Linking) <────────── Task 8
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
```

### Model Options

- `gpt-4o` (default) - Best quality
- `gpt-4-turbo` - Faster, slightly lower quality
- `gpt-3.5-turbo` - Fastest, lower quality

## Notes

1. **API Costs**: Each trial requires approximately 10-20 LLM calls
2. **Processing Time**: Single trial takes about 25-60 seconds
3. **Result Quality**: Depends on the completeness of ClinicalTrials.gov data
4. **Version History**: Task 7 requires R with `cthist` package installed

## Related Modules

- `agents/visualization/` - CTKG visualization utilities
- `examples/ctkg_agent_demo/` - Demo scripts and example outputs

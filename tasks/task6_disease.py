"""
Task 6: Mapping significant Task 5 findings to disease/symptom-level impact.

Logic:
1) Read Task 5 `kg_triples` (new display structure).
2) Keep only statistically significant triples.
3) Group evidence by the same head pair (head_entity_1, head_entity_2).
4) Single-step LLM inference per head-pair group:
   - infer affected disease/symptoms
   - assess clinical meaningfulness
   - choose relation context
5) Emit disease-level triples in dual-head form when comparator exists.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import re

from .base import BaseTaskHandler, TaskResult

logger = logging.getLogger(__name__)

_ALLOWED_CONTEXT_RELATIONS = {
    "treat",
    "improve",
    "alleviate",
    "prevent",
    "reduce_risk",
    "no_clinical_effect",
    "worsen",
}


DISEASE_MEANINGFULNESS_PROMPT = """
# Role
You are a clinical inference and evidence evaluation engine.

# Task
Given statistically significant comparative outcome evidence from one trial arm/group pair,
infer what disease(s) or symptom(s) these outcomes actually reflect and determine
whether each inferred candidate is clinically meaningful.

# Trial Context
- Trial ID: {nct_id}
- Registered Conditions: {registered_conditions}
- Trial Title: {trial_title}
- Brief Summary: {brief_summary}

# Group Context
- Head Group 1: {head_entity_1}
- Head Group 2: {head_entity_2}
- Comparative Relation from statistics: {comparative_relation}

# Outcome Evidence (all significant outcomes for this same head pair)
{outcome_evidence_json}

# Allowed relation context labels
- treat
- improve
- alleviate
- prevent
- reduce_risk
- no_clinical_effect
- worsen

# Instructions
1. Use trial design and outcome definitions to infer disease/symptom targets.
2. Do not just copy registered conditions when outcome semantics imply a different target.
3. Prefer standardized medical terms.
4. If uncertain, lower confidence.
5. For each inferred disease candidate, independently decide clinical meaningfulness and assign relation context.

# Output JSON
Return ONLY valid JSON:
{{
  "inferred_diseases": [
    {{
      "disease_name": "...",
      "disease_type": "disease" | "symptom" | "syndrome" | "biomarker" | "risk_factor",
      "inference_confidence": "HIGH" | "MEDIUM" | "LOW",
      "inference_reasoning": "...",
      "linked_outcomes": ["outcome core measurement or title"],
      "is_clinically_meaningful": true | false,
      "context_relation": "treat" | "improve" | "alleviate" | "prevent" | "reduce_risk" | "no_clinical_effect" | "worsen",
      "effect_direction": "beneficial" | "harmful" | "neutral",
      "magnitude": "large" | "moderate" | "small" | "minimal",
      "clinical_relevance": "...",
      "evidence_strength": "STRONG" | "MODERATE" | "WEAK",
      "limitations": ["..."],
      "reasoning": "..."
    }}
  ],
  "primary_affected_condition": "...",
  "outcome_clinical_domain": "...",
  "is_same_as_registered": true | false,
  "clinical_interpretation": "..."
}}
"""


class Task6DiseaseMapping(BaseTaskHandler):
    @property
    def task_name(self) -> str:
        return "task6_disease_mapping"

    def execute(
        self,
        trial_data: Any,
        task5_results: Dict = None,
        **kwargs,
    ) -> TaskResult:
        try:
            if task5_results is None:
                return self._create_error_result(["Task 5 results required for disease mapping"])

            task5_triples = task5_results.get("kg_triples", []) or []
            significant_triples = [
                t for t in task5_triples
                if (t.get("relation_attributes", {}) or {}).get("is_significant") is True
            ]

            if not significant_triples:
                return self._create_success_result({
                    "disease_relations": [],
                    "kg_triples": [],
                    "message": "No statistically significant Task 5 triples. Task 6 skipped.",
                    "summary": {
                        "task5_total_triples": len(task5_triples),
                        "task5_significant_triples": 0,
                        "head_pair_groups": 0,
                        "clinically_meaningful_relations": 0,
                    },
                })

            context = self.prepare_input(trial_data)
            grouped = self._group_by_head_pair(significant_triples)

            disease_relations: List[Dict[str, Any]] = []
            kg_triples: List[Dict[str, Any]] = []
            inference_details: List[Dict[str, Any]] = []
            errors: List[str] = []

            for group_key, group_bundle in grouped.items():
                try:
                    inferred = self._infer_group_disease_and_meaningfulness(group_bundle, context)
                    if not inferred or "error" in inferred:
                        errors.append(f"Disease inference failed for head pair {group_key}")
                        continue

                    inference_details.append({
                        "head_pair": list(group_key),
                        "inferred_conditions": inferred.get("inferred_diseases", []),
                        "is_same_as_registered": inferred.get("is_same_as_registered"),
                        "clinical_interpretation": inferred.get("clinical_interpretation"),
                    })

                    for disease_info in inferred.get("inferred_diseases", []) or []:
                        if not isinstance(disease_info, dict):
                            continue
                        rel = str(disease_info.get("context_relation", "")).strip().lower()
                        if rel not in _ALLOWED_CONTEXT_RELATIONS:
                            disease_info["context_relation"] = "improve"
                        if not disease_info.get("is_clinically_meaningful", False):
                            continue

                        triple = self._build_group_disease_triple(
                            trial_data.nct_id,
                            group_bundle,
                            disease_info,
                            inferred,
                        )
                        kg_triples.append(triple)
                        disease_relations.append(self._build_legacy_relation_record(triple, inferred, disease_info))

                except Exception as exc:
                    logger.error("Task6 group processing failed: %s", exc)
                    errors.append(str(exc))

            return self._create_success_result({
                "disease_relations": disease_relations,
                "kg_triples": kg_triples,
                "inference_details": inference_details,
                "summary": {
                    "task5_total_triples": len(task5_triples),
                    "task5_significant_triples": len(significant_triples),
                    "head_pair_groups": len(grouped),
                    "clinically_meaningful_relations": len(kg_triples),
                    "unique_inferred_diseases": len({t.get("tail") for t in kg_triples if t.get("tail")}),
                },
                "errors": errors,
            })

        except Exception as exc:
            logger.error("Task 6 failed: %s", exc)
            return self._create_error_result([str(exc)])

    def prepare_input(self, trial_data: Any) -> Dict[str, Any]:
        conditions = getattr(trial_data, "conditions", {}) or {}
        description = getattr(trial_data, "description", {}) or {}
        identification = getattr(trial_data, "identification", {}) or {}

        return {
            "nct_id": trial_data.nct_id,
            "registered_conditions": conditions.get("conditions", []) or [],
            "brief_summary": description.get("briefSummary", "") or "",
            "title": identification.get("officialTitle", identification.get("briefTitle", "")) or "",
        }

    def _group_by_head_pair(self, significant_triples: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for triple in significant_triples:
            h1 = triple.get("head_entity_1") or ""
            h2 = triple.get("head_entity_2") or ""
            key = (h1, h2)

            if key not in grouped:
                grouped[key] = {
                    "head_entity_1": h1,
                    "head_entity_2": h2,
                    "head_entity_1_attributes": triple.get("head_entity_1_attributes") or {},
                    "head_entity_2_attributes": triple.get("head_entity_2_attributes") or {},
                    "comparative_relation": (triple.get("relation") or {}).get("comparative_relation"),
                    "outcome_evidence": [],
                    "source_triples": [],
                }

            bundle = grouped[key]
            if not bundle.get("comparative_relation"):
                bundle["comparative_relation"] = (triple.get("relation") or {}).get("comparative_relation")

            tail_attrs = triple.get("tail_attributes")
            tail_attr_obj = {}
            if isinstance(tail_attrs, list) and tail_attrs:
                tail_attr_obj = tail_attrs[0] if isinstance(tail_attrs[0], dict) else {}
            elif isinstance(tail_attrs, dict):
                tail_attr_obj = tail_attrs

            rel_attrs = triple.get("relation_attributes") or {}
            outcome_evidence = {
                "tail": triple.get("tail"),
                "core_measurement": tail_attr_obj.get("core_measurement"),
                "original_title": tail_attr_obj.get("original_title"),
                "outcome_type": (tail_attr_obj.get("attributes") or {}).get("outcome_type"),
                "paramType": (tail_attr_obj.get("attributes") or {}).get("paramType"),
                "unitOfMeasure": (tail_attr_obj.get("attributes") or {}).get("unitOfMeasure"),
                "timeFrame": (tail_attr_obj.get("attributes") or {}).get("timeFrame"),
                "description": (tail_attr_obj.get("attributes") or {}).get("description"),
                "p_value": rel_attrs.get("p_value"),
                "statistical_analysis": rel_attrs.get("statistical_analysis"),
                "statistical_conclusion": rel_attrs.get("statistical_conclusion"),
                "group_values": rel_attrs.get("group_values"),
                "context": (triple.get("relation") or {}).get("context"),
            }
            bundle["outcome_evidence"].append(outcome_evidence)
            bundle["source_triples"].append(triple)

        return grouped

    def _infer_group_disease_and_meaningfulness(
        self,
        group_bundle: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        prompt = DISEASE_MEANINGFULNESS_PROMPT.format(
            nct_id=context.get("nct_id", ""),
            registered_conditions=json.dumps(context.get("registered_conditions", []), ensure_ascii=False),
            trial_title=context.get("title", "")[:400],
            brief_summary=context.get("brief_summary", "")[:1200],
            head_entity_1=group_bundle.get("head_entity_1", ""),
            head_entity_2=group_bundle.get("head_entity_2", "") or "N/A",
            comparative_relation=group_bundle.get("comparative_relation") or "N/A",
            outcome_evidence_json=json.dumps(group_bundle.get("outcome_evidence", []), ensure_ascii=False, indent=2),
        )
        response = self.call_llm(prompt)
        return self._parse_response(response)

    def _build_group_disease_triple(
        self,
        nct_id: str,
        group_bundle: Dict[str, Any],
        disease_info: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        relation_context = disease_info.get("context_relation") or "improve"
        comparative_relation = group_bundle.get("comparative_relation")

        return {
            "head_entity_1": group_bundle.get("head_entity_1"),
            "head_entity_2": group_bundle.get("head_entity_2") or None,
            "relation": {
                "comparative_relation": comparative_relation,
                "context": relation_context,
            },
            "tail": disease_info.get("disease_name"),
            "head_entity_1_type": "Group",
            "head_entity_2_type": "Group" if group_bundle.get("head_entity_2") else None,
            "tail_type": "Disease/symptoms",
            "head_entity_1_attributes": group_bundle.get("head_entity_1_attributes") or {},
            "head_entity_2_attributes": group_bundle.get("head_entity_2_attributes") or {},
            "tail_attributes": {
                "linked_outcomes": disease_info.get("linked_outcomes", []),
            },
            "relation_attributes": {
                "evidence_quality": {
                    "strength": disease_info.get("evidence_strength"),
                    "limitations": disease_info.get("limitations", []),
                },
                "reasoning": disease_info.get("reasoning", ""),
            },
            "provenance": {
                "nct_id": nct_id,
            },
        }

    def _build_legacy_relation_record(
        self,
        triple: Dict[str, Any],
        inference_result: Dict[str, Any],
        disease_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        h1_attrs = triple.get("head_entity_1_attributes") or {}
        intervention_name = h1_attrs.get("group_title") or triple.get("head_entity_1")
        return {
            "intervention": intervention_name,
            "inferred_disease": triple.get("tail"),
            "disease_type": (triple.get("tail_attributes") or {}).get("disease_type", "disease"),
            "relation": (triple.get("relation") or {}).get("context"),
            "relation_qualifier": (triple.get("relation") or {}).get("comparative_relation"),
            "inference_confidence": (triple.get("tail_attributes") or {}).get("inference_confidence", "MEDIUM"),
            "is_same_as_registered": inference_result.get("is_same_as_registered", False),
            "clinical_domain": (triple.get("tail_attributes") or {}).get("outcome_clinical_domain"),
            "is_clinically_meaningful": disease_info.get("is_clinically_meaningful", False),
            "evidence_quality": (triple.get("relation_attributes") or {}).get("evidence_quality", {}),
            "source_outcome": [
                o.get("core_measurement")
                for o in (triple.get("relation_attributes", {}) or {}).get("outcome_support", [])
                if o.get("core_measurement")
            ],
            "nct_id": (triple.get("provenance") or {}).get("nct_id", ""),
        }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r"\{[\s\S]*\}", response or "")
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON response: %s", exc)

        return {
            "error": "Failed to parse response",
            "raw_response": (response or "")[:500],
        }

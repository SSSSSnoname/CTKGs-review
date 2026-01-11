"""
Task Handlers Package

Contains implementations for all 9 CTKG construction tasks.
"""

from .base import BaseTaskHandler, TaskResult
from .task1_outcome import Task1OutcomeStandardization
from .task2_intervention import Task2InterventionProfiling
from .task3_eligibility import Task3EligibilityStructuring
from .task4_purpose import Task4PurposeInference
from .task5_statistical import Task5StatisticalConclusions
from .task6_disease import Task6DiseaseMapping
from .task7_dynamic import (
    Task7DynamicCTKG,
    VersionNode,
    VersionChangeRelation,
    FieldChange,
    ChangeType,
    VersionFieldExtractor,
    VersionDiffEngine,
    TextFieldStructurer,
    TerminationNormalizer,
    TerminationInfo,
    TRACKED_FIELDS,
    TRACKED_FIELDS_CONFIG,
    TERMINATION_CATEGORIES,
)
from .task8_assembly import Task8CTKGAssembly
from .task9_linking import Task9EntityLinking

__all__ = [
    'BaseTaskHandler',
    'TaskResult',
    'Task1OutcomeStandardization',
    'Task2InterventionProfiling',
    'Task3EligibilityStructuring',
    'Task4PurposeInference',
    'Task5StatisticalConclusions',
    'Task6DiseaseMapping',
    'Task7DynamicCTKG',
    # Task 7 additional exports
    'VersionNode',
    'VersionChangeRelation',
    'FieldChange',
    'ChangeType',
    'VersionFieldExtractor',
    'VersionDiffEngine',
    'TextFieldStructurer',
    'TerminationNormalizer',
    'TerminationInfo',
    'TRACKED_FIELDS',
    'TRACKED_FIELDS_CONFIG',
    'TERMINATION_CATEGORIES',
    # Task 8-9
    'Task8CTKGAssembly',
    'Task9EntityLinking',
]


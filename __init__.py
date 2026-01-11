"""
Construction Agent Package

This package contains modules for extracting structured information from clinical trial data
and performing statistical analysis to construct knowledge graph triplets.

Main Components:
- CTKGConstructionAgent: Main agent class for orchestrating all tasks
- Task modules (1-9): Individual task implementations
- Data loaders: API and file-based data loading
"""

# Main agent
from .ctkg_agent import (
    CTKGConstructionAgent,
    TaskType,
    CTKGType,
    TrialData,
    TaskResult,
    CTKGTriple,
    create_agent,
    process_trial
)

__all__ = [
    # Main agent
    'CTKGConstructionAgent',
    'TaskType',
    'CTKGType',
    'TrialData',
    'TaskResult',
    'CTKGTriple',
    'create_agent',
    'process_trial',
]

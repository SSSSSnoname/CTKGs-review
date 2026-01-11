"""
Base Task Handler

Abstract base class for all task handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Container for task execution results."""
    task_name: str
    success: bool
    data: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class BaseTaskHandler(ABC):
    """
    Abstract base class for CTKG construction task handlers.
    
    All task handlers should inherit from this class and implement
    the execute() method.
    """
    
    def __init__(
        self,
        llm_client=None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        """
        Initialize the task handler.
        
        Args:
            llm_client: Pre-configured LLM client
            api_key: API key for LLM service
            model: Model name to use
        """
        self.llm_client = llm_client
        self.api_key = api_key
        self.model = model
        
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return the name of this task."""
        pass
    
    @abstractmethod
    def execute(self, trial_data: Any, **kwargs) -> TaskResult:
        """
        Execute the task on the given trial data.
        
        Args:
            trial_data: TrialData object containing parsed trial information
            **kwargs: Additional task-specific parameters
            
        Returns:
            TaskResult containing the execution results
        """
        pass
    
    @abstractmethod
    def prepare_input(self, trial_data: Any) -> Dict:
        """
        Prepare input data for the task.
        
        Args:
            trial_data: TrialData object
            
        Returns:
            Dictionary containing prepared input data
        """
        pass
    
    def call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLM response as string
        """
        if self.llm_client:
            return self._call_with_client(prompt, **kwargs)
        elif self.api_key:
            return self._call_with_openai(prompt, **kwargs)
        else:
            raise ValueError("No LLM client or API key configured")
    
    def _call_with_client(self, prompt: str, **kwargs) -> str:
        """Call LLM using pre-configured client."""
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    def _call_with_openai(self, prompt: str, **kwargs) -> str:
        """Call LLM using OpenAI API directly."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    def _create_success_result(self, data: Dict, metadata: Optional[Dict] = None) -> TaskResult:
        """Create a successful task result."""
        return TaskResult(
            task_name=self.task_name,
            success=True,
            data=data,
            metadata=metadata or {}
        )
    
    def _create_error_result(self, errors: List[str], partial_data: Optional[Dict] = None) -> TaskResult:
        """Create an error task result."""
        return TaskResult(
            task_name=self.task_name,
            success=False,
            data=partial_data or {},
            errors=errors
        )


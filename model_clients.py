"""
Model client adapters for CTKG construction.

This module adds a local Hugging Face client that mimics the minimal
OpenAI chat.completions interface used by task handlers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _CompletionResponse:
    choices: List[_Choice]


class _CompletionsAdapter:
    def __init__(self, parent: "LocalHFChatClient"):
        self._parent = parent

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> _CompletionResponse:
        prompt = self._parent._build_prompt(messages)
        content = self._parent._generate(prompt, **kwargs)
        return _CompletionResponse(choices=[_Choice(message=_Message(content=content))])


class _ChatAdapter:
    def __init__(self, parent: "LocalHFChatClient"):
        self.completions = _CompletionsAdapter(parent)


class LocalHFChatClient:
    """
    Minimal OpenAI-like chat client backed by a local Hugging Face model.

    Requires `transformers` and optionally `torch` at runtime.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
        **pipeline_kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.pipeline_kwargs = pipeline_kwargs

        self._pipe = None
        self._tokenizer = None
        self.chat = _ChatAdapter(self)

    def _ensure_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "LocalHFChatClient requires transformers. Install with: pip install transformers torch"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        task = self.pipeline_kwargs.pop("task", "text-generation")
        self._pipe = pipeline(
            task=task,
            model=self.model_path,
            tokenizer=self._tokenizer,
            device_map=self.device if self.device != "cpu" else None,
            **self.pipeline_kwargs,
        )

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        self._ensure_pipeline()
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _generate(self, prompt: str, **kwargs) -> str:
        self._ensure_pipeline()

        max_new_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_new_tokens))
        temperature = kwargs.pop("temperature", self.temperature)
        do_sample = kwargs.pop("do_sample", self.do_sample)

        outputs = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False,
        )
        if not outputs:
            return ""

        first = outputs[0]
        if isinstance(first, dict):
            return first.get("generated_text", "") or first.get("text", "")
        return str(first)

